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

# different directions
dif_dirs = [
    "ai",
    "physical",
    "economics",
    "quantum",
    "math"
]

# processing
try:
    for d in dif_dirs:
        search = arxiv.Search(
            query=d,
            max_results=2000,
            sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()
        res = client.results(search)

        file_name = d+".txt"
        with open(root_path + "/dataset/" + file_name, "w") as f:
            print("The summary of the total collection for " + d +" is " + str(len(list(res))))
            for r in client.results(search):
                f.write(r.summary + '\n' + '\n')
    print("collection successful")
except:
    print("collect failed")