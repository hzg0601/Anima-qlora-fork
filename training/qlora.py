# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""train函数的流程：
1. 初始化各组命令行参数，将其组合为一个命名空间
    1.1 调用 transformers.HfArgumentParser，将model,data,train,generate
        的Arguments组合为一个命名空间
    1.2 调用HfArgumentParser.parse_args_into_dataclasses
        将命令行参数解析为指定数据类类型的实例
    1.3 调用transformers.GenerationConfig将generate参数实例
        转换为generation_config实例，并赋值给model.generation_config
        命名空间
    1.4 将model,data,train的参数实例组合为一个整的的命名空间
2. 调用get_last_checkpoint检测是否完成训练，如未完成，返回checkpoint的路径
3. 调用get_accelerate_model执行多卡部署，调用print_trainable_parameters打印可训练参数
    3.1 检测设备数量，最大显存，设定量化时梯度更新的数据类型compute_dtype
    3.2 调用AutoModel类的from_pretrained方法加载模型
        1. 其中device_map="auto",设定的torch基本数据为torch.float32或torch.bfloat16
        2. 并额外设置了quantization_config字段，该字段的值是一个BitsAndBytesConfig类
            该类中可设定load_in_8bit，llm_int8_threshold,
            并额外添加了bnb_4bit_compute_type,bnb_4bit_use_double_quant,
            bnb_4bit_quant_type等字段用于双重量化
        3. 模型加载后设定model_parallel,is_parallelizable为True
        4. 对于非full_finetune情况，调用prapare_model_for_kbit_training再次
            初始化模型
        5. 按模型配置，启动gradient_checkpointing_enable
        6. 若checkpint_dir路径下没有adapter路径，调用PeftModel.from_pretrained模型，
            加载lora权重，合并权重，更新模型；如果没有adapter路径，则调用find_all_linear_names
            找出除lm_head外的所有线性层，调用LoraConfig配置lora的config,配置对象为上述线性层，
            而后基于config和原生模型调用get_peft_model初始化peft模型
        7. 最后将模型的lora,lm_head,embed_tokens层转换为bf16或fp32,
            将norm层转换为fp32.
4. 设置tokenizer.
    4.1 AutoTokenizer.from_pretrained方法实例化tokenizer,
    4.2 调用smart_tokenizer_and_embedding_resize更新模型的词表长度和新加入的token的权重
    4.3 对于llama类模型，加入eos_token,bos_token,unk_token.
5. 调用组装数据函数make_data_module，加载数据，分割数据,返回一个包括：
    train_dataset,eval_dataset,predict_dataset,datacollator的字典。
    5.1 定义数据加载函数load_data,用于读取不同的数据集，如果是本地数据，
        则调用local_dataset读取指定格式的数据，读取数据类为dataset的Dataset类，
        支持.json,.jsonl,.csv,.tsv,读取后直接调用Dataset类的train_test_split
        进行数据分割。
    5.2 定义数据格式函数format_dataset，将不同来源的数据(由dataset_format字段标识)
        统一为具有input,output字段的字典
        1. 如果是alpaca类数据，直接调用extract_alpaca_dataset函数进行处理，该函数
            提供了一个模板将数据自带的instruction,input统一为alpaca的prompt作为输入。
        2. 如果是chip2类数据，则数据text字段值的格式为
            "$input:\n<bot> :<human> :\n<bot> :$output""，将数据抽取写入input,output
            对应字段即可。
        3. 如果是self-instruct类数据，则仅需进行字段重命名即可。
        4. 如果是hh-rlhf类数据，则input字段为'',output为样例的chosen字段
        5. 如果是oasst1类数据，则input字段为'',output为样例的text字段
        6. 处理完毕后，将dataset中的其他字段删除。
    5.3 调用load_data函数加载数据，如果是在debug模式下，训练只使用前200条数据
    5.4 调用format_dataset函数，处理数据格式
    5.5 如果是执行eval或predict任务，但数据集没有eval字段，对train数据集进行分割
        生成eval数据集。若执行train任务，取出数据即可。
    5.6 若规定了max_eval_samples字段，则只取规定数量的样本进行评测，
        如果存在group_by_length字段，则计算每个样本输入输出长度之和，计入length字段下。

    5.7 实例化组装数据类DataCollatorForCausalLM，该类拥有如下属性：
        tokenizer: transformers.PreTrainedTokenizer
        source_max_len: int
        target_max_len: int
        train_on_source: bool
        predict_with_generate: bool
        定义的__call__方法的流程如下：
        1. 将输入的instances: Sequence[Dict]的input,output分别取出作为sources,targets;
        2. 调用实例化的tokenizer对输入的sources,targets进行tokenize;
        3. 如果不执行predict_with_generate,则将输入输出拼接作为一条输入
            如果不执行train_on_source,则将输入掩码，以掩码和输出作为一条label,
            否则，就把输入和输出拼接作为一条label（train_on_source除了目标文本之外是否还对输入进行训练）
          如果执行predict_with_generate,则只将输入作为一条输入;
          得到input_ids和labels列表。
        4. 对input_ids和labels进行padding,输出以B*T(最长句子的长度)的形式。
        5. 基于tokenizer.pad_token_id对input_ids进行掩码得到attention_mask,
            将input_ids,attention_mask,labels作为字段返回一个字典。
6. 实例化Seq2SeqTrainer类进行训练，Seq2SeqTrainer类是Trainer的子类，
   该类的参数包括model,tokenizer,args(training_args),**kwargs.
7. 对于非full_finetune任务，向Seq2SeqTrainer实例增加callback类SavePeftModelCallback
    该类是transformers.TrainerCallback的子类，定义了save_model,on_save,on_train_end
    等方法，用于保存peft模型的权重，更新文件处理时间，生成completed文件夹。
8. 对于sample_generate任务，向Seq2SeqTrainer实例增加callback类SampleGenerateCallback，
    该类是transformers.TrainerCallback的子类，定义了on_evaluate方法，对给定的
    样本问题进行生成式回答。
9. 如果执行mmlu_eval，则加载对应的数据，分割数据，实例化评估准则，定义MMLUEvalCallback类，
    并加入trainer的callback类中，支持的数据包括mmlu/zero-shot,mmlu/five-shot两类数据
    ?定义abcd_idx列表对，A,B,C,D进行tokenize.
    1. MMLUEvalCallback
10.  统计每个数据类型下的参数的个数和比例
11. 对train/eval/predict任务，调用trainer对应的train/evaluate/predict方法，
    记录并更新train/evaluate/predict的表现，对于predict任务，还需要将输出decode,
    将预测的输出、输出+输入写入文件
12. 将最终的metrics写入本地文件。


"""
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
# 数据类可以被认为是“具有默认值的可变命名元组”提供了一个类装饰器，它使用 PEP 526“变量注释的语法”中定义的类型注释来检查变量的类定义。
#  在本文档中，此类变量称为字段。 使用这些字段，装饰器将生成的方法定义添加到类中以支持
# 实例初始化、repr、比较方法以及可选的其他方法，如规范部分中所述。 
# 这样的类称为数据类，但该类实际上没有什么特别之处：装饰器将生成的方法添加到类中并返回给定的类。
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
# bitandbytes raise RuntimeError: CUDA Setup failed despite GPU being available.
# 需要根据初始打印信息中CUDA SETUP: CUDA runtime path found: /home/huangzhiguo/anaconda3/lib/libcudart.so
# 找到其指向的CUDA version，若指向的不是目标cuda version
# 则修改libcudart.so软链接指向的libcudart.so版本，先删除libcudart.so，再通过命令建立软链接
# ln -s libcudart.so.11.2.72 libcudart.so

# 某些情况下用pip install bitsandbytes命令只能下载0.38.1，此时需要从pypi下载whl文件，选择从文件安装

# 如果报出RuntimeError: Something when wrong when trying to find file. Maybe you do not have a linux system?
# 先检查CUDA SETUP: Loading binary /home/huangzhiguo/anaconda3/envs/transformers/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda112.so
# libbitsandbytes_cuda112.so是否正确

# 如果正确则可能是__main__.py函数出错
import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


torch.backends.cuda.matmul.allow_tf32 = True

logging_file_path = f"./qlora_logs.log"

handlers = [
    logging.FileHandler(logging_file_path),
    logging.StreamHandler(sys.stdout)
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="timdettmers/guanaco-33b-merged"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='Belle_0.5M',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
# bf16是transformers.Seq2SeqTrainingArguments的方法
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    #? fp4和nf4是什么？-> float point 4, normal float 4
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    ) 
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    # Wandb是Weights & Biases的缩写，是类似TensorBoard， visdom的一款可视化工具,是属于Python的，不是Pytorch的

    # wandb是最大的特点是能自动上传云端，让你即使在外面或者旅行途中也能随时随地查看模型崩没崩。甚至觉得某个模型跑的不好了在手机上就可以停掉它。

    # 但是wandb必须要联网，没网的本地电脑是没法跑的
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    sample_generate: bool = field(default=False, metadata={"help": 'If do sample generation on evaluation.'})
    debug_mode: bool = field(default=True, metadata={"help": 'debug mode sample 200 train/eval samples for validation'})
    deepspeed: str = field(default="./deepspeed_config.json",metadata={"help": "the path to deepspeed config file"})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 

# 该函数表明，所有的量化层都使用了bnb.nn.Linear4bit或bnb.nn.Linear8bit
def find_all_linear_names(args, model):
    """找出所有线性量化层的层名，lm_head层除外"""
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 找出所有的量化层，只把模块的最后一个层级加到lora_modlue_names中
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# 模型生成的callback类，针对样例数据和样例的prompt，生成样例的response.
class SampleGenerateCallback(transformers.TrainerCallback):
    "A callback that prints a sample generations of the model in the process of training"

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("on_evaluate in SampleGenerateCallback...")
        sample_inputs = [
            '用一句话描述地球为什么是独一无二的。',
            '中国是否应该推出刺激政策救楼市？',
            '如何更好地融入新工作圈子'
        ]
        if "model" in kwargs:
            for sample_input in sample_inputs:
                tokenizer = kwargs['tokenizer']
                inputs = "Below is an instruction that describes a task. " \
                         "Write a response that appropriately completes the request.\n\n" \
                         "### Instruction:\n{sample_input}\n\n### Response: ".format(sample_input=sample_input)
                logger.info(f"sample input: {inputs}")
                model = kwargs['model']
                input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
                input_ids = input_ids.to('cuda')
                generation_output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=70,
                )
                #print(generation_output)
                logger.info(f"sample output: {tokenizer.decode(generation_output[0])}")

        else:
            logger.info(f"model not found in kwargs, skipping")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                # os.utime() 方法用于设置指定路径文件最后的修改和访问时间。在Unix,Windows中有效。
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

# 使用accelerate库多卡部署模型
def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    # 如果执行full finetune，则保存位数必须是16或32位，即必须是float32,bf16,fp16
    if args.full_finetune: assert args.bits in [16, 32]

    logger.info(f'loading base model {args.model_name_or_path}...')
    # 量化时更新梯度
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    # transformers的每个模型仓库都包括如下几个文件：
    # 1. configuration_{model}.py，该脚本定义一个PretrainedConfig子类，用于指定模型的配置字段->config_dict；
    # 2. config.json, 指定architectures字段，以及模型结构和token的字段。
    # 3. modeling_{model}.py
    # 4. 模型检查点
    # 5. tokenizer_{model}.py
    # 6. tokenizer_config.json
    
    # kwargs可用于更新configuration对象（在加载后）并启动模型（例如，output_attentions=True）。 
    # 根据是提供config还是自动加载而表现不同：
    # 如果提供了config，**kwargs 将直接传递给底层模型的 __init__ 方法
    # 如果未提供config，kwargs 将首先传递给Config类初始化函数（from_pretrained()）。
    # 与config属性对应的 kwargs 的每个键将用于覆盖所述属性,不对应任何config属性的剩余键将传递给底层模型的 __init__ 函数。

    # AutoModel.from_pretrained流程：
    #    1 kwargs如果存在config,trust_remote_code字段，则将其从kwargs中pop取出；
    #    2 定义hub_kwargs_names列表，从kwargs中pop出hug类字段到hub_kwargs中；
    #    3 如果config不是PretrainedConfig类，则调用AutoConfig.from_pretrained类生成config,kwargs;
    #       (1). AutoConfig.from_pretrained首先在kwargs中写入_from_auto，name_or_path,trust_remote_code字段的值
    #       (2). 然后调用PretrainedConfig.get_config_dict方法，返回config_dict为模型的配置信息，kwargs去除了加载和下载配置
    #           该方法调用_get_config_dict,_get_config_dict方法从kwargs中取出下载配置，如cache_dir,resume_download关键字等
    #           如果方法传入的name_or_path是文件路径，则设定resolved_config_file为该路径，is_local=True
    #           如果name_or_path经补全后为url地址，则调用download_url下载resolved_config_file
    #           否则，从kwargs取出_config_file或默认为config.json,调用cached_file
    #           加载或下载resolved_config_file,然后调用_dict_from_json_file加载config_dict.
    #*           最后将Config类实例config_dict和kwargs返回，故config_dict为模型的配置信息，kwargs去除了加载和下载配置。
    #*           

    #       （3).如果"auto_map"在config_dict字典中，且"AutoConfig"在"auto_map"的值中，则设定has_remote_code为True;
    #            如果"model_type"在config字典中，且"model_type"的值在CONFIG_MAPPING_NAMES字典中，则设定has_local_code为True ;

    #       (4).如果has_remote_code和trust_remote_code的值均为True,则调用get_class_from_dynamic_module加载模型类：
    #           该方法根据传入的class_reference， pretrained_model_name_or_path
    #           参数注册对应的类.py文件，从py文件加载类。get_class_from_dynamic_module传入的class_ref
    #           在auto_map字段下的AutoConfig字段中。class_reference存在两种格式:
    #           "sgugger/my-bert-model--modeling.MyBertModel"，或modeling.MyBertModel
    #           然后调用get_cached_module_file函数加载文件，调用get_class_in_module加载对应的类，
    #           加载类完成后，调用类的from_pretrained方法实例化Config类。
    #       (5) 如果"model_type"在config_dict中，则取出model_type的值，调用_LazyConfigMapping类
    #           基于CONFIG_MAPPING_NAMES初始化，使用importlib.import_module和getattr加载类，
    #           再调用类的from_dict方法基于config_dict实例化Config类。
    #       (6) 如果没有"auto_map","model_type"字段，则根据name_or_path在CONFIG_MAPPING中
    #           中进行匹配，再调用from_dict方法基于config_dict实例化Config类。
    #*      故AutoConfig类的from_pretrained方法会返回模型的Config实例config_dict,和去除下载和加载配置信息的kwargs.
    #   4. 如果实例config_dit中存在auto_map字段，取出实例的auto_map值，取出前调用函数将auto_map的值统一加上repo_id--，
    #       然后取出__name__的值，得到repo_id--{model}的字符串，然后get_class_from_dynamic_module函数，
    #       该函数将下载或加载模型类对应的脚本，在通过get_class_in_module函数调用importlib.import_module
    #       方法加载对应的模型类，模型类通过from_pretrained方法接受name_or_path,额外的--位置--模型参数model_args,
    #       AutoConfig类实例config,hub_kwargs,kwargs进行实例化，故hub_kwargs,kwargs是_BaseAutoModelClass的属性
    #   5. 如果AutoConfig子类的实例config是_model_mapping的key,则调用_get_model_class函数加载模型类，该函数
    #       接受config，模型类字典_model_mapping{模型类：模型类格式或格式列表}，如果模型子类仅有一个格式，
    #       则直接返回模型类；否则构造{类名：类}字典，从config的architectures字段取出其值，
    #       按照值在{类名：类}字典中取出对应的格式类，然后调用from_pretrained方法实例化模型；
    #
    #* quantization_config参数是在PretrainedModel类定义的，自定义的类或从architectures加载的类都继承了PretrainedModel类
    #* 具体支持的参数，参考modeling_utils.py文件
    # AutoModelForCausalLM.from_pretrained也支持enable_deepspeed，enable_fsdp等

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map='auto',
        max_memory=max_memory,
        # BitsAngBytesConfig(load_in_8bit=False,llm_int8_threshold=6,llm_int8_skip_modules,
        # llm_int8_enable_fp32_cpu_offload,**kwargs)

        # llm_int8_threshold: float=6,对应于论文LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
        # 中离群值阈值。高于这个阈值的隐状态值会被视为离群值，在这些值上的参数会以fp16格式进行。
        # 隐状态值通常是正态分布的，范围在[-35,3.5]之间，但有一些特殊的系统异常值，它们在大型模型中的分布非常不同。
        # 这些异常值通常位于 [-60, -6] 或 [6, 60] 区间内。Int8 量化适用于值幅度约为 5，超过5的话，
        # int8量化后的表现将极大下降。一个好的默认阈值是 6，但对于更不稳定的模型（小模型、微调），可能需要较低的阈值。

        # llm_int8_skip_modules: 不进行int8量化的层列表，对于CausalLM类，lm_head通常会保持原始类型。
        # 而某些模型如Jukebox，在不同的位置也有多个head，因此也不能进行量化。
        
        # llm_int8_enable_fp32_cpu_offload, 此标志用于了解此功能的高级用例和用户。 
        # 如果你想将你的模型分成不同的部分并在 GPU 上以 int8 运行一些部分而在 CPU 上以 fp32 运行一些部分，
        # 你可以使用这个标志。 这对于卸载大型模型（例如 `google/flan-t5-xxl`）很有用。 
        # 请注意，int8 操作不会在 CPU 上运行。

        # kwargs: 初始化configuration对象的其他参数。
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
    )
    # 如果梯度计算的类型为torch.float16,量化位数为4，则检验设备的计算能力，如果计算上限是8，则设备支持bf16格式
    # 提示用户可以启用--bf16参数
    if compute_dtype == torch.float16 and args.bits == 4:
        # get_device_capability, 返回设备的计算能力，8以上支持bf16格式
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    #? 重复了吧，AutoModelForCausalLM.from_pretrained已经设置了这个参数，且完全可以将之设为一个变量
    #?为什么要升类型，混合精度计算里，更新梯度用torch.float32,而前向传播用torch.float16,是否与此有关
    #? 若如此，则compute_dtype，就是指前向传播的数据类型
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    # 
    if not args.full_finetune:
        # 1. prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
        # 该方法封装了训练前准备模型的所有协议，包括：
        # (1) 将layernorm层投射为fp32;
        # (2) 将所有的输出embedding层设为require_grads
        # (3) 将lm_head向上转换为fp32.
        # 具体方案：
        # (1) 将基础模型的所有参数设为requires_grad = False
        # (2) 将所有的fp16或bf16的参数设为pf32
        # (3) 如果基础模型is_loaded_in_{k}bit，且use_gradient_checkpointing=True,
        #     则enable_input_require_grads
        # (4) 启动模型的gradient_checkpointing_enable
    #* -----与下文也重复了-----------
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    #* ------------------------------
    if not args.full_finetune:
        if checkpoint_dir is not None:
            # 如果checkpoint_dir存在，则从adapter_model中加载adapter权重，返回一个PeftModel类
            logger.info("Loading adapters from checkpoint.")
            # Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            # 如果checkpoint不存在，则调用find_all_linear_names
            # 找出所有线性量化层的层名，lm_head层除外
            # 用LoraConfig实例化为LoraConfig类，module对象为lm_head外所有线性层
            # 再调用get_peft_model实例化一个PeftModel类
            logger.info(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            # LoraConfig是PeftConfig的一个子类
            # LoraConfig(
            # r: dimension of LoRA
            # lora_alpha:The alpha parameter for Lora scaling.
            # lora_dropout: The dropout probability for Lora layers.
            # fan_in_fan_out:Set this to True if the layer to replace stores weight like (fan_in, fan_out).
            # bias:
            # modules_to_save: List of modules apart from LoRA layers to be set as trainable
            #                  and saved in the final checkpoint.
            # layers_to_transform: The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            #                      the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            #                      transformations on the layer at this index.
            # layers_pattern: The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            #                 pattern is not in the common layers pattern.
            # peft_type=PeftType.LORA,
            # task_type:
            # name_or_path        
            # revision:The specific model version to use
            # inference_mode: Whether to use inference mode
            # target_modules:
            # init_lora_weights)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # Returns a Peft model object from a model and a config.
            # 返回一个PeftModel类
            model = get_peft_model(model, config)
    # 对于模型的所有模块，如果模块是一个LoraLayer，且支持bf16类型，则将该层转换为bf16类型
    # 如果模块是一个正则层，则模块的类型转换为float32位
    # 如果模块是一个lm_head层或embed_tokens层，且支持bf16类型，则将该层转换为bf16类型
    #* 因此可以看出，模型的基本类型为fp32或bf16,
    #* 转换类型的方式是：如果支持bf16，则lora层，lm_head,embed_tokens层需要转换为bf16，否则保持为fp32
    #* 这意味着，lora,lm_head,embed_tokens层需要的是更大的表示范围
    #* 而norm层无论基本类型是fp32还是bf16，都需要转换为fp32，表明norm层需要更高的精度。
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # numel，返回数组中元素的个数
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    # ! use trainable twice may be slightly confusing
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable ratio: {100 * trainable_params / all_param}"
    )

# tokenizer调用add_special_tokens将自定义的special_tokens_dict加入tokenizer，更新token词表
# 然后模型调用resize_token_embeddings更新模型的词表长度
# 计算旧embedding的均值，然后新加入的token的权重设为旧embedding的均值
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 数据处理类：__call__方法的输入为一个dict的序列,dict包括两个键：input, output
# 1. 取出输入instances的input和output作为sources和targets
# 2. 执行tokenize操作，并执行truncation操作，保证输出的tokenize不超过max_len
# 3. 组装数据
#    如果不是predict_with_generate，就只以source为输入，且不设labels
#    如果是predict_with_generate,则以source+target为输入，
#       如果执行train_on_source,则以source+target为label,
#       如果不执行train_on_source,则以source掩码+target为label.
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize，max_length和truncation配和设定token后序列的最大长度
        # 也可设定truncation为longest_first,only_first，only_second,False'
        # longest_first会逐个token进行截断，如果输入是一对句子，函数会从最长的句子中移除一个token
        # only_first与longest_first基本一致，只是如果输入是一对句子，函数只会对第一个句子进行截断
        # only_second,则是只对第二句子进行截断，
        # False则不会进行截断，从而可能输出超出模型允许的最大长度的句子。
        # truncated和padding的choices基本一致

        # tokenizer的返回值为BatchEncoding类，它包含如下字段：
        # input_ids,token_typed_ids,attention_mask,overflow_tokens,
        # num_truncated_tokens,special_tokens_mask,length
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = [] 
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'], 
            tokenized_targets['input_ids']
        ):
            # 如果不执行predict_with_generate,则将输入输出拼接作为一条输入
            #   如果不执行train_on_source,则将输入掩码，以掩码和输出作为一条label
            #   否则，就把输入和输出拼接作为一条label
            # 如果执行predict_with_generate,则只将输入作为一条输入
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        # batch_first=True,则输出以B*T(最长句子的长度)的形式，否则以T*B的形式
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # 如果不执行predict_with_generate,则没有labels
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        # ne=not_equal,
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
    
# 将examples中instances字段，reformulations字段的每个样本组装到out的input,output字段中
def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    """
    examples: Dict[str,Sequence[Dict[str,str]]]
    """
    out = {
        'input': [], 
        'output': [],
    }
    # examples:DataCollatorForCausalLM输入的容器
    for example_instances in examples['instances']:
        for instance in example_instances:
            # out['input']相当于prompt,out['output']相当于label
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    # 如果数据还进行了reformulation,则examples还会有reformulations字段
    # 对于每个reformulation字段下的example_reformulations，只有它不为None
    # 则将它的prompt添加入out['input']中，将它的输出添加入out['output']中
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out
# alpaca的prompt模板，一种没有输入，一种有输入
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}
# 如果样例有输入，则使用prompt_input模板构造输入的prompt，
# 否则使用prompt_no_input构造输入的prompt
def extract_alpaca_dataset(example):
    """
    example:Dict[str,str]
    """
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}
# 加载并分割数据
def local_dataset(dataset_name):
    if dataset_name.endswith('.json'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(filename=dataset_name, format='jsonlines')
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")
    
    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset
#! 重复下载数据
def recursive_load_dataset(dataset_name,max_try=20):
    #for some regions and countries, `load_dataset` casually raise ConnectionError, 
    # so recursively downloading the dataset would make the project more robust.
    try_turn = 0
    while True:
        try:
            data = load_dataset(dataset_name) 
            return data
        except Exception as e:
            logger.info("Download dataset failed, re-downloading...")
            try_turn += 1
            if try_turn > max_try:
                logger.info("The number of retries exceeded `max_try`,maybe you are offline, please check the network.")
                raise ConnectionError
            else:
                continue
            
#! 如果增加新的数据集，应在此处适配。首先弄清模型的输入形式，然后弄清数据集的格式
def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples   
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return recursive_load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return recursive_load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return recursive_load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return recursive_load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return recursive_load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return recursive_load_dataset("akoksal/LongForm")
        # 包含了多语言，数据质量较高
        elif dataset_name == 'oasst1':
            return recursive_load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        # guanaco_belle_merge_v1.0 数据质量一般
        # bigscience/xP3 中文数据质量一般，xP3all中文质量稍好，中文数据集下有不少藏语实例
        elif dataset_name == 'chinese-vicuna':
            return recursive_load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")
        elif dataset_name == "Belle_0.5M":
            return recursive_load_dataset("BelleGroup/train_0.5M_CN")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "alpaca"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        # alpaca类数据集的格式应类似一个dataframe,包括input,instruction两个字段
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or 
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            # 输出的是以input为key,prompt为字段的字典，故format_dataset的输出也是如此
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            # chip2的格式类似一个Series,每个元素都是一个字典，字典有text字段
            # text字段值的格式为"$input:\n<bot> :<human> :\n<bot> :$output""
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            # self-instruct数据集的格式为
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset
        
     # Load dataset.
    dataset = load_data(args.dataset)
    # 在debug模式下，训练只使用前200条数据
    if args.debug_mode:
        dataset['train'] = dataset['train'].filter(lambda x,i: i < 200, with_indices=True)
        #dataset['eval'] = dataset['eval'].filter(lambda x,i: i < 200, with_indices=True)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            logger.info('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        # 如果存在group_by_length字段，则计算每个样本输入输出长度之和，计入length字段下
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    # 组装数据类的实例化
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer, 
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None, 
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
# 检查训练是否完成，如未完成返回最新一步的检查点路径和未完成训练的flag值
def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        # 如果存在checkpoint_dir下的complete文件路径，则表明训练完成，返回None,True；否则返回None,False

        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        # 如果不存在checkpoint_dir路径，则表面训练未完成，遍历checkpoint_dir，
        # 找出最新一步，如果最新一步为0，表明刚刚开始训练，返回None,False
        # 如果最新一步不为0，则返回最新一步的文件路径和False.
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    # HfArgumentParser是ArgumentParser的一个子类，它使用dataclasses的类型提示来生成arguments
    # 该类需要与原生argparse一起使用，允许在parser在初始化以后增加（非dataclass支持）arguments,
    # 在parsing后，该类将作为一个额外的命名空间返回
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    # 将命令行参数解析为指定数据类类型的实例。 这依赖于 argparse 的“ArgumentParser.parse_known_args”。
    # parse_args_into_dataclasses(self,args=None,return_remaining_strings=False,look_for_args_file=True,
    #    args_filename=None,args_file_flag=None)
    # return_remaining_strings:如果为真，还返回剩余参数字符串的列表。
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # 将generation_args作为training_args的generation_config字段
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # 将所有args组装作为一个完整的args
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    logger.info(f"args: {args}")
    # 检测是否完成训练
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.info('Detected that training was already completed!')
    # 多卡部署model
    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    logger.info('loaded model')
    set_seed(args.seed)

    # Tokenizer
    # 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True if "pythia" in args.model_name_or_path or "gpt-neox" in args.model_name_or_path else False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
    )
    # 如果tokenizer存在_pad_token,则调用smart_tokenizer_and_embedding_resize，其逻辑为：
    # tokenizer调用add_special_tokens将自定义的special_tokens_dict加入tokenizer，更新token词表
    # 然后模型调用resize_token_embeddings更新模型的词表长度
    # 计算旧embedding的均值，然后新加入的token的权重设为旧embedding的均值
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # 对于llama族的LlamaTokenizer（如guanaco），需要增加eos_token,bos_token,unk_token
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary. 
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        logger.info('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(                    
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    # 实例化组装数据类，加载数据，分割数据，
    # 返回一个包括train_dataset,eval_dataset,predict_dataset,datacollator的字典
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    # 实例化Seq2SeqTrainer类进行训练，Seq2SeqTrainer类是Trainer的子类
    # Seq2SeqTrainer(model,args,data_collator,train_dataset,eval_dataset,tokenizer,
    # model_init,computer_metrics,callbacks,optimizers,preprocess_logit_for_metric)
    # 该类的train方法直接继承自Trainer,重写了evaluate，predict，prediction_step方法
    # evaluate和predict方法基于max_length,args.max_new_tokens更新了max_length参数，
    # 基于num_beams和args.generation_num_beams更新了num_beams参数
    
    # todo 如果需要使用deepspeed，则需要在training_args中加入deepspeed参数
    # todo 参数指向deepspeed配置文件的位置
    # todo deepspeed配置的参数参考https://www.deepspeed.ai/docs/config-json/
    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    # 增加callback类
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.sample_generate:
        trainer.add_callback(SampleGenerateCallback)
    # 如果执行mmlu_eval，则加载对应的数据，分割数据，实例化评估准则，
    # 定义MMLUEvalCallback类，并加入trainer的callback类中
    if args.do_mmlu_eval:
        # zs->zeroshot
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': '../data/mmlu/zero_shot_mmlu_val.json',
                'test': '../data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': '../data/mmlu/five_shot_mmlu_val.json',
                'test': '../data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes.
    # 统计每个数据类型下的参数的个数和比例
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        #! TypeError: not all arguments converted during string formatting
        logger.info(f"the type, number, and ratio of data are {k, v, v/total},respectively")
    #? all_metrics 记录metric的字典
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not. 
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        # prediction的output中pad_token_id的预测值为-100
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        # 转换为token_id对应的str
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # 将预测的输出、输出+输入写入文件，记录预测的metric
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        logger.info(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)
    # 将最终的metric写入文件
    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
