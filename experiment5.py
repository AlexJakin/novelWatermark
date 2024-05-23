'''
HarryPotter小说实验
'''
from model.waterMarked import waterMarked
from model.LLM import LLM
from dataModel.NovelDataProcess import NovelDataProcess
import random
import os
from log.runLog import MyLogging

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

# 记录日志
logger = MyLogging("mylog",file=root_path+"/testProject/out/log/experiment5.log")

EPOCH = 3
MODEL_NAME = 'openai-community/gpt2'
# MODEL_NAME = 'openai-community/gpt2-xl'
INIT_SENT = 'This is a sentence from a science fiction novel.' #  前提引导


base_llm = LLM.LLM(MODEL_NAME)

for i in range(EPOCH):
    # Generate two LLMs, one watermarked and one not
    seed = random.randint(1, 9999)
    llm = waterMarked.WatermarkedLLM(base_llm, seed)
    dataset = NovelDataProcess(root_path + "/testProject/dataset/HarryPotter.txt", INIT_SENT)

    dataset_len = len(dataset)

    wk_evals = []

    for j, data in enumerate(dataset[:5000]):
        prompt = data['prompt']
        human_response = data['rest']
        rest_len = len(human_response)

        #  生成带水印文本
        id_list = llm.llm.str_to_idlist(prompt)
        generate_text_with_wk = llm.generate_next_ids(id_list, gen_len=rest_len, wk=True)
        text_with_wk = generate_text_with_wk[len(id_list):]

        # 测试模型是否正确分类带水印的文本
        # 带水印攻击的，会一直干扰拉低logit 导致最后结果低于没有带水印的
        with_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=True)
        with_no_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=False)
        wk_eval = (with_no_wk_total_log_prob > with_wk_total_log_prob )
        wk_evals.append(wk_eval)

        #print(f'The result of the HarryPotter experiment: Iteration {j + 1}/{dataset_len}: WaterMark: {wk_eval}', flush=True)
        print(f'The result of the HarryPotter experiment: Iteration {j + 1}/{5000}: WaterMark: {wk_eval}', flush=True)


    # 计算分类精度
    eval_correct = sum(wk_evals) / len(wk_evals) * 100

    print(f'The result of the HarryPotter experiment: Watermarked text correctly classified: {eval_correct:.2f}%')
    # 将结果写入日志
    logger.info(f'The result of the HarryPotter experiment: Watermarked text correctly classified: {eval_correct:.2f}%')