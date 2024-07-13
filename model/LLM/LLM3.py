# from transformers import pipeline, set_seed
#
# set_seed(32)
# generator = pipeline('text-generation', model="facebook/opt-1.3b", do_sample=True, num_return_sequences=5)
# generator("The man worked as a")
#
# from datasets import load_dataset
#
# # English only
# en = load_dataset("allenai/c4", "en")
#
# # Other variants in english
# en_noclean = load_dataset("allenai/c4", "en.noclean")
# en_noblocklist = load_dataset("allenai/c4", "en.noblocklist")
# realnewslike = load_dataset("allenai/c4", "realnewslike")
#
# # Multilingual (108 languages)
# multilingual = load_dataset("allenai/c4", "multilingual")
#
# # One specific language
# es = load_dataset("allenai/c4", "es")

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo')
# tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
assert tokenizer.encode('hello world') == [15339, 1917]

