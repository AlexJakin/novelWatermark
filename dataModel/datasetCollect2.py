'''
1、 ai
2、physical
3、economics
4、quantum
5、math
'''
import os
import arxiv

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

try:
    print("collection start------")
    # one direction for llm
    llm_search = arxiv.Search(
        query="llm",
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance
    )
    # create an api client
    client = arxiv.Client()
    res = client.results(llm_search)
    # processing
    with open(root_path+"/dataset/llm_abstracts.txt", "w") as f:
        print("The summary of the total collection is " + str(len(list(res))))
        for r in client.results(llm_search):
            print("\033[0;31;40m the paper title：\033[0m" + r.title)
            print("\033[0;32;40m the paper summary：\033[0m" + r.summary)
            print("\n")
            f.write(r.summary + '\n' + '\n')
    print("collection successful")
except:
    print("collect failed")