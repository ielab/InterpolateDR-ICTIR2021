# the file will need one input files, which is the entry file location (where the testing dataset are allocated,
# which contain the normalised scores)
import os
from tqdm import tqdm
import sys
entry_file = sys.argv[1]
print("1")
bm25 = open(entry_file+'bm25.txt', 'r')
print("2")
bert = open(entry_file+'bert.txt', 'r')

print("3")
Linesbm25 = bm25.readlines()
bm25.close()
Linesbert = bert.readlines()

if not os.path.exists(entry_file + "output"):
    os.mkdir(entry_file + "output")
    os.mkdir(entry_file + "eval")


for alphaindex in tqdm(range(0,11)):
    alpha = alphaindex/10
    bertv = 1-alpha
    output = open(entry_file + 'output/output'+str(alpha)+'.txt', 'w')
    pairs = {}
    min_line = len(Linesbm25)
    index = int(min_line/1000)
    index_bert = int(len(Linesbert)/1000)
    current_counter = 0
    previous = ""
    qid_list_bert = [n for n in range(0,index_bert)]

    while current_counter!=len(Linesbm25):
        bm25_set = {}
        bert_set = {}
        output_set = {}
        start_counter = -1
        qid = Linesbm25[current_counter].split()[0]
        while Linesbm25[current_counter].split()[0] == qid:
            bm25line = Linesbm25[current_counter]
            bm25score = float(bm25line.split()[2])
            did_1 = bm25line.split()[1]
            bm25_set[did_1] = bm25score
            current_counter = current_counter + 1
            if current_counter == len(Linesbm25):
                break
        for i in range(0,index_bert):
            if Linesbert[i*1000].split()[0] == qid:
                qid_list_bert.remove(i)
                start_counter = i*1000
        if start_counter == -1:
            continue
        for i in range(start_counter, start_counter+1000):
            bertline = Linesbert[i]
            did_2 = bertline.split()[1]
            bertscore = float(bertline.split()[2])
            bert_set[did_2] = bertscore
            if did_2 in bm25_set.keys():
                output_set[did_2] = (alpha * bm25_set.get(did_2)) + (bertv * bertscore)
            else:
                output_set[did_2] = bertv * bertscore
                #print(str(qid) +  " " +str(did_2))

        for key in bm25_set.keys():
            if key not in bert_set.keys():
                output_set[key] = alpha * bm25_set.get(key)
        count = 0
        for sort_k in sorted(output_set, key=output_set.get, reverse=True):
            output.write(qid + " Q0 " + str(sort_k) + " " + str(count+1) + " " + str(output_set.get(sort_k)) + " ielab-uniCOIL\n")
            count += 1
            if count == 1000:
                break
        bm25_set.clear()
        bert_set.clear()
        output_set.clear()
    for kkk in qid_list_bert:
        k_start = kkk*1000
        k_end = kkk*1000+1000
        count = 1
        for line in Linesbert[k_start:k_end]:
            items = line.split()
            qid = items[0]
            pid = items[1]
            score = items[2]
            output.write(
                qid + " 0 " + str(pid) + " " + str(count + 1) + " " + score + " fused \n")
            count = count+1

