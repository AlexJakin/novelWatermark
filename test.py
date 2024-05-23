import torch

from model.LLM import LLM
from model.waterMarked import waterMarked

model_name = "openai-community/gpt2"
llm = LLM.LLM(model_name=model_name)

SEED = 12345
test_str = 'The weather is bad today, so'
token_list = llm.str_to_tokenlist(test_str)
#print(token_list) # ['The', 'Ġweather', 'Ġis', 'Ġbad', 'Ġtoday']

#id_list = llm.tokenlist_to_idlist(token_list)
#print(id_list) # tensor([ 464, 6193,  318, 2089, 1909])

# id_list = llm.tokenlist_to_idlist(token_list)
# print(f'id list: {id_list}')
# token_list = llm.idlist_to_tokenlist(id_list)
# print(f'token list: {token_list}')
# out_str = llm.tokenlist_to_str(token_list)
# print(f'Final output: {out_str}')

# idlist = torch.IntTensor([235, 2, 3, 4, 5])
# print(idlist)
# print(llm.get_logits(idlist).size()) # torch.Size([1, 5, 50257])

# 普通测试
idlist = llm.tokenlist_to_idlist(token_list)
logits = llm.get_logits(idlist)[0, -1, :]
probs = llm.logits_to_probs(logits)
out_id = llm.decode(probs)
input_ids = torch.cat((idlist, out_id.unsqueeze(0)))

# print(logits.size()) # torch.Size([1, 7, 50257])
# print(logits[0, -1, :].size()) # torch.Size([50257])
#print(out_id) # 下一个单词的id
# print(llm.idlist_to_tokenlist(input_ids)) # ['The', 'Ġweather', 'Ġis', 'Ġbad', 'Ġtoday', ',', 'Ġso', 'Ġwe']

# for i, id_val in enumerate(input_ids):
#     print(i, id_val)
wllm = waterMarked(llm, SEED)
prob_w_watermark = wllm.eval_log_prob(input_ids)
print(prob_w_watermark)