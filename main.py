# test
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

'''
1、将文本分词为词语 (token) 序列
'''
# text = "time flies like an arrow"
text = "china"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

'''
2、pytorch加载嵌入层
'''
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

'''
3、将每一个词语转换为对应的词向量
'''
inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())