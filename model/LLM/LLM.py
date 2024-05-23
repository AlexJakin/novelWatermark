import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        # because the task is text generation, use the AutoModelForCausalLM to construct model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    # use top_k
    def decode(self, probs, top_k=-1):
        if top_k > 0:
            ids = torch.argsort(probs, descending=True)[:top_k]
            probs = probs[:top_k]
        else:
            ids = torch.tensor([i for i in range(self.vocab_size)])
        # choose the highest prob
        return random.choices(ids , weights=probs)[0]

    # convert sentence into id list
    def str_to_idlist(self, input_str):
        token_list = self.str_to_tokenlist(input_str)
        return self.tokenlist_to_idlist(token_list)

    # convert id into token
    def idlist_to_tokenlist(self, id_list):
        return self.tokenizer.convert_ids_to_tokens(id_list)

    # convert token into id
    def tokenlist_to_idlist(self, token_list):
        return torch.tensor(self.tokenizer.convert_tokens_to_ids(token_list))

    # split input data to token
    def str_to_tokenlist(self, input_str):
        return self.tokenizer.tokenize(input_str)

    # convert token to sentence
    def tokenlist_to_str(self, token_list):
        return self.tokenizer.convert_tokens_to_string(token_list)

    # get the logit of next possible tokens
    def get_logits(self, id_list):
        with torch.no_grad():
            # Add a batch dimension if necessary
            if len(id_list.shape) < 2:
                id_list = id_list.unsqueeze(0) # 最外层加一个维度
            outputs = self.model(id_list)
            logits = outputs.logits
        return logits

    # get a probability distribution given logits
    def logits_to_probs(self, logits, temp=1):
        return torch.nn.functional.softmax(logits/temp, dim=-1)