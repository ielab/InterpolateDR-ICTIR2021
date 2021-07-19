#the use of this document is to count unjudged documents in experimental needs.
import tqdm
import os

folder = os.getcwd()+'/'
year = "2020"
input_query = folder + "Deepdl/" + year + "_qrels/dev.qrels.tsv"
input_folder = folder + "Deepdl/RepBert_BM25/" + year + "/"
input_folder2 = folder + "Deepdl/ANCE_BM25/" + year + "/"
alphalist = [0,5,10]
query_dict = {}
query_lines = open(input_query, 'r').readlines()

for line in query_lines:
    items = line.split()
    if items[0] not in query_dict.keys():
        query_dict[items[0]] = [items[-2]]
    else:
        current_l = query_dict.get(items[0])
        current_l.append(items[-2])
        query_dict[items[0]] = current_l
Rep_bm25 = {}
for alphaindex in alphalist:
    alpha = alphaindex/10
    output_lines = open(input_folder + 'output/output'+str(alpha)+'.txt', 'r').readlines()
    count = 0
    for line in output_lines:
        items = line.split()
        if items[0] in query_dict.keys():
            if items[2] not in query_dict.get(items[0]):
                count = count+1

    Rep_bm25[alpha] = count


ANCE_bm25 = {}
for alphaindex in alphalist:
    alpha = alphaindex/10
    output_lines = open(input_folder2 + 'output/output'+str(alpha)+'.txt', 'r').readlines()
    count = 0
    for line in output_lines:
        items = line.split()
        if items[0] in query_dict.keys():
            if items[2] not in query_dict.get(items[0]):
                count = count+1
    ANCE_bm25[alpha] = count
print("rep_bm25")
print(Rep_bm25)
print("\n")
print("ance_bm25" )
print(ANCE_bm25)