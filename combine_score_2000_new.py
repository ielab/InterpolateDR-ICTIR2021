# the file will need one input files, which is the entry file location (where the testing dataset are allocated,
# which contain the normalised scores)
import os
from tqdm import tqdm
import sys
entry_file = sys.argv[1]
print("1")
bm25 = open(entry_file+'bm25.txt', 'r')
Linesbm25 = bm25.readlines()
lenbm25 = len(Linesbm25)
bm25.close()


bert = open(entry_file+'bert.txt', 'r')
print("3")


alpha = 0.3
bertv = 1 - alpha

if not os.path.exists(entry_file + "output"):
    os.mkdir(entry_file + "output")
    os.mkdir(entry_file + "eval")
output = open(entry_file + 'output/output' + '0.3.txt', 'w')
current_counter = 0
for readlineindex in tqdm(range(0,int(277144000/1000))):
    Linesbert = []
    for localindex in range(0,1000):
        Linesbert.append(bert.readline())
    pairs = {}
    bm25_set = {}
    bert_set = {}
    output_set = {}
    start_counter = 0
    qid = Linesbm25[current_counter].split()[0]
    if qid != Linesbert[0].split()[0]:
        continue

    while Linesbm25[current_counter].split()[0] == qid:
        bm25line = Linesbm25[current_counter]
        bm25score = float(bm25line.split()[2])
        did_1 = bm25line.split()[1]
        bm25_set[did_1] = bm25score
        current_counter = current_counter + 1
        if current_counter == lenbm25:
            break

    for i in range(0, 1000):
        bertline = Linesbert[i]
        did_2 = bertline.split()[1]
        bertscore = float(bertline.split()[2])
        bert_set[did_2] = bertscore
        if did_2 in bm25_set.keys():
            output_set[did_2] = (alpha * bm25_set.get(did_2)) + (bertv * bertscore)
        else:
            output_set[did_2] = bertv * bertscore
            # print(str(qid) +  " " +str(did_2))
    for key in bm25_set.keys():
        if key not in bert_set.keys():
            output_set[key] = alpha * bm25_set.get(key)
    count = 0

    for sort_k in sorted(output_set, key=output_set.get, reverse=True):
        output.write(
            qid + " 0 " + str(sort_k) + " " + str(count + 1) + " " + str(output_set.get(sort_k)) + " fused \n")
        count += 1
        if count == 1000:
            break
    if current_counter == lenbm25:
        break
    bm25_set.clear()
    bert_set.clear()
    output_set.clear()
    Linesbert.clear()
bert.close()
output.close()
