import os
from torch.utils.data import Dataset, DataLoader
from nltk import sent_tokenize

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

class NovelDataProcess(Dataset):
    def     __init__(self, data_path, init_sentence):
        self.data = []
        file = open(data_path, "r")
        novels = file.read().split("\n")[:-1]

        for nl in novels:
            nl.replace(u'\u3000',u'')  # 去掉空格
            sentences = sent_tokenize(nl)  # Split into different sentences
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

if __name__ == '__main__':
    init = ("This is a sentence from a science fiction novel.")
    dataset = NovelDataProcess(root_path + "/dataset/HarryPotter.txt", init)
    print(len(dataset))
    print(dataset[0])
    print(len(dataset[0]['rest']))
