import os
from torch.utils.data import Dataset, DataLoader
from nltk import sent_tokenize

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

class dataProcess(Dataset):
    def __init__(self, data_path, init_sentence):
        self.data = []
        file = open(data_path, "r")
        paper_abstracts = file.read().split("\n\n")[:-1]
        # print(paper_abstracts)

        for ab in paper_abstracts:
            sentences = sent_tokenize(ab)  # Split into different sentences
            if len(sentences) >= 3:
                prompt = init_sentence + ' ' + sentences[0]
                rest = ' '.join(sentences[1:])
                datapoint = {
                    'prompt': prompt,
                    'rest': rest
                }
                self.data.append(datapoint)

    # return length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
# if __name__ == '__main__':
#     init = ("Potential harms of large language models can be mitigated by watermarking model output, embedding signals "
#             "into generated text that are invisible to humans.")
#     dataset = dataProcess(root_path + "/dataset/llm_abstracts.txt", init)
#     print(len(dataset))
