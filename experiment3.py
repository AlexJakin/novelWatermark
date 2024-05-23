'''
识别arxiv不同方向的摘要人写和机器写 统一生成100个单词
'''
from model.waterMarked import waterMarked
from model.LLM import LLM
from dataModel.dataProcess import dataProcess
import random
import os
from log.runLog import MyLogging

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

# 记录日志
logger = MyLogging("mylog",file=root_path+"/testProject/out/log/experiment3.log")

EPOCH = 3
MODEL_NAME = 'openai-community/gpt2'
INIT_SENT = 'The following sentences are taken from the abstract of a scientific paper.' #  前提引导
# different directions
DIF_DIRS = [
    "ai",
    "physical",
    "economics",
    "quantum",
    "math"
]


base_llm = LLM.LLM(MODEL_NAME)

for i in range(len(DIF_DIRS)):
    seed = random.randint(1, 9999)
    llm = waterMarked.WatermarkedLLM(base_llm, seed)
    dataset = dataProcess(root_path + "/testProject/dataset/" + DIF_DIRS[i] + ".txt", INIT_SENT)

    dataset_len = len(dataset)

    human_evals = []

    for j, data in enumerate(dataset):
        prompt = data['prompt']
        human_response = data['rest']

        # 测试我们的模型是将在arxiv数据集归类为人类还是人工智能生成 log_prob大的是添加了水印的gpt写的
        dir_whole_abstract = prompt + ' ' + human_response
        id_list = llm.llm.str_to_idlist(dir_whole_abstract)
        marked_total_log_prob = llm.calc_log_prob(id_list, with_wk=True)  # gpt写的，即做了水印标记
        unmarked_total_log_prob = llm.calc_log_prob(id_list, with_wk=False)  # 人类写的logit
        current_human_eval_status = (unmarked_total_log_prob < marked_total_log_prob)  # 记录计算后判断成功
        human_evals.append(current_human_eval_status)
        print(f'The result of the {DIF_DIRS[i]} direction: Iteration {j + 1}/{dataset_len}: Human: {current_human_eval_status}', flush=True)


    human_eval_correct = sum(human_evals) / len(human_evals) * 100

    print(f'The result of the {DIF_DIRS[i]} direction: Human-generated text correctly classified: {human_eval_correct:.2f}%')
    # 将结果写入日志
    logger.info(f'The result of the {DIF_DIRS[i]} direction: Human-generated text correctly classified: {human_eval_correct:.2f}%')