'''
HarryPotter Novel experiment TPR 100%
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
logger1 = MyLogging("mylog1",file=root_path+"/testProject/out/log/experiment5_success.log")
logger2 = MyLogging("mylog2",file=root_path+"/testProject/out/log/experiment5_error.log")

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

    for j, data in enumerate(dataset[:500]):
        prompt = data['prompt']
        human_response = data['rest']
        rest_len = len(human_response)

        id_list = llm.llm.str_to_idlist(prompt)
        generate_text_with_wk = llm.generate_next_ids(id_list, gen_len=rest_len, wk=True)
        marked_text_content = llm.llm.tokenlist_to_str(llm.llm.idlist_to_tokenlist(generate_text_with_wk))
        text_with_wk = generate_text_with_wk[len(id_list):]

        with_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=True)
        with_no_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=False)
        wk_eval = (with_no_wk_total_log_prob > with_wk_total_log_prob )
        wk_evals.append(wk_eval)

        #print(f'The result of the HarryPotter experiment: Iteration {j + 1}/{dataset_len}: WaterMark: {wk_eval}', flush=True)
        print(f'The result of the HarryPotter experiment: Iteration {j + 1}/{500}: WaterMark: {wk_eval}', flush=True)
        if wk_eval == True:
            logger1.info(
                f'Iteration {j + 1}/{500} The prompt: {prompt},//======//, the generate: {marked_text_content}, the with_wk_total_log_prob is {with_wk_total_log_prob}, the with_no_wk_total_log_prob is {with_no_wk_total_log_prob}')
        else:
            logger2.info(
                f'Iteration {j + 1}/{500} The prompt: {prompt},//======//, the generate: {marked_text_content}, the with_wk_total_log_prob is {with_wk_total_log_prob}, the with_no_wk_total_log_prob is {with_no_wk_total_log_prob}')


    eval_correct = sum(wk_evals) / len(wk_evals) * 100

    print(f'The result of the HarryPotter experiment: Watermarked text correctly classified: {eval_correct:.2f}%')
    logger.info(f'The result of the HarryPotter experiment: Watermarked text correctly classified: {eval_correct:.2f}%')