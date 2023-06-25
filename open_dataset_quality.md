# 开源中文LLM数据集

1. `timdettmers/openassistant-guanaco` 包括多种语言，中文数据的质量似乎不错。
2. `bigscience/xP3,xP3all` xP3 中文数据质量一般，xP3all中文质量稍好，中文数据集下有不少藏语实例。
3. `Chinese-Vicuna/guanaco_belle_merge_v1.0`中文数据质量一般，target普遍过短，且似乎审核不严。`Chinese-Vicuna/instruct_chat_50k.jsonl`数据质量尚可，但存在需要过滤的数据项。
4. `zzzzhhh/LLaMa-zn`,`c-s-ale/alpaca-gpt4-data-zh`等中文数据是基于chatgpt生成的，质量似乎不错，output普遍比较丰富，。
5. `bavest/fin-llama-dataset`是英文数据集，而且数据质量不高
6. `facebook/flores`包含200多种语言，官方维护应该比较可靠
7. `mteb/sts22-crosslingual-sts`包含多种语言，但似乎是对文档相似度打标，中文以新闻为主。
8. `QingyiSi/Alpaca-CoT`用GPT-4生成，包含多种语言，以CoT为instruct.
9. `BelleGroup/train_0.5M_CN`数据质量较好，但似乎没有统一的模板，也存在output较短的问题。multiturn_chat_0.8M 
10. `TigerResearch/sft_zh`具有比较丰富的数量来源，如果知乎、alpaca中文等，质量似乎也不错。

