
from model.waterMarked import waterMarked
from model.LLM import LLM
from dataModel.dataProcess import dataProcess
import random
import os
from log.runLog import MyLogging

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)


logger = MyLogging("mylog",file=root_path+"/testProject/out/log/experiment4.log")

EPOCH = 3
#MODEL_NAME = 'openai-community/gpt2'
#MODEL_NAME = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
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
    # Generate two LLMs, one watermarked and one not
    seed = random.randint(1, 9999)
    llm = waterMarked.WatermarkedLLM(base_llm, seed)
    dataset = dataProcess(root_path + "/testProject/dataset/" + DIF_DIRS[i]  + ".txt", INIT_SENT)

    dataset_len = len(dataset)

    wk_evals = []

    for j, data in enumerate(dataset):
        prompt = data['prompt']
        #human_response = data['rest']


        id_list = llm.llm.str_to_idlist(prompt)
        generate_text_with_wk = llm.generate_next_ids(id_list, gen_len=100, wk=True)
        text_with_wk = generate_text_with_wk[len(id_list):]


        with_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=True)
        with_no_wk_total_log_prob = llm.calc_log_prob(generate_text_with_wk, with_wk=False)
        wk_eval = (with_no_wk_total_log_prob > with_wk_total_log_prob )
        wk_evals.append(wk_eval)

        print(f'The result of the {DIF_DIRS[i]} direction: Iteration {j + 1}/{dataset_len}: WaterMark: {wk_eval}', flush=True)



    eval_correct = sum(wk_evals) / len(wk_evals) * 100

    print(f'The result of the {DIF_DIRS[i]} direction: Watermarked text correctly classified: {eval_correct:.2f}%')
    logger.info(f'The result of the {DIF_DIRS[i]} direction: Watermarked text correctly classified: {eval_correct:.2f}%')