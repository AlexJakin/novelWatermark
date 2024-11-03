from model.LLM import LLM
import torch
import math
import random
import numpy as np

class WatermarkedLLM:
    def __init__(self, llm, seed):
        self.llm = llm
        self.alpha = 0.15
        self.sqare_logitSeq  = 0.2
        self.window_size = 20
        self.window_nums = 0
        self.sparse_mark = []
        self.generate_watermark(seed)

    def generate_watermark(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        wk = torch.rand((self.llm.vocab_size,), generator=g)
        self.watermark = torch.nn.functional.softmax(wk, dim=-1)

    def applyWK(self, probs):
        return ((1 - self.alpha) * probs) + (self.alpha * self.watermark)

    def generate_window_size(self, seq_len):
        self.window_nums = seq_len // self.window_size

    def generate_sparse_mark(self, gen_len):
        sp_mark = np.zeros(gen_len)
        self.window_nums = gen_len // self.window_size
        if self.window_nums > 2:
            for i in range(self.window_nums - 2):
                ids = random.randint(0, 19)
                self.sparse_mark[(i + 1) * self.window_size + ids] = 1
        else:
            sp_mark[gen_len//2] = 1
        self.sparse_mark = sp_mark

    def generate_next_ids(self, prompt_ids, gen_len = 30, wk=True):
        for i  in range(gen_len):
            next_logits = self.llm.get_logits(prompt_ids)[0, -1, :]
            next_probs = self.llm.logits_to_probs(next_logits)
            if wk:
                next_probs = self.applyWK(next_probs)
            next_ids = self.llm.decode(next_probs)
            prompt_ids = torch.cat((prompt_ids, next_ids.unsqueeze(0)))
        return prompt_ids

    def generate_next_ids_win(self, prompt_ids, gen_len=30, wk=True):
        # generate a sparse mark
        self.generate_sparse_mark(gen_len)
        for i in range(gen_len):
            next_logits = self.llm.get_logits(prompt_ids)[0, -1, :]
            next_probs = self.llm.logits_to_probs(next_logits)
            if wk and self.sparse_mark[i]:
                next_probs = self.applyWK(next_probs)
            next_ids = self.llm.decode(next_probs)
            prompt_ids = torch.cat((prompt_ids, next_ids.unsqueeze(0)))

        return prompt_ids


    def calc_log_prob(self, ids, with_wk=True):
        logits = self.llm.get_logits(ids)[0, :-1, :]
        total_log_prob = 0
        for i, id in enumerate(ids):
            if i != 0:
                probs = self.llm.logits_to_probs(logits[i - 1, :])
                if with_wk:
                    nextOne_prob = ((1 - self.alpha) * probs[id] + self.alpha * self.watermark[id])
                else:
                    nextOne_prob = probs[id]
                log_prob = math.log(nextOne_prob)
                total_log_prob -= log_prob

        return total_log_prob


if __name__ == '__main__':
    # model_name = 'openai-community/gpt2'
    # model_name = 'openai-community/gpt2-xl'
    model_name = 'facebook/opt-1.3b'

    llm = LLM.LLM(model_name)
    seed = 12345
    wllm = WatermarkedLLM(llm, seed)

    test_str = 'The weather is bad today, so '
    token_list = llm.str_to_tokenlist(test_str)
    id_list1 = llm.tokenlist_to_idlist(token_list)
    id_list2 = wllm.generate_next_ids(id_list1)

    #print(wllm.watermark.size()) # torch.Size([50257])

    token_list = llm.idlist_to_tokenlist(id_list2)
    out_str = llm.tokenlist_to_str(token_list)

    print(f'Final output: {out_str}')