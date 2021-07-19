# this file is used to count the number of relevant documents in each level, which is shown in the experiment paper.

from tqdm import tqdm
import os
base_dir = os.getcwd()+"/"

entry_file = base_dir + "Deepdl/RepBert_BM25/2019/" #modify this for different dataset folder
query_file = base_dir+ "Deepdl/2019_qrels/" + "dev.qrels.tsv" # modify this to different query file location

output_file = entry_file + "output/" # this is for output location
out_file = entry_file + "distribution.txt"
bm25file = output_file+"output1.0.txt"
bertfile = output_file+"output0.1.txt"

query_line = open(query_file, 'r').readlines()
out = open(out_file, 'w')
relevance_dict_0 = {}
relevance_dict_1 = {}
relevance_dict_2 = {}
relevance_dict_3 = {}
qid_set = set()




for line in tqdm(query_line):
    qid = line.split()[0]
    did = line.split()[2]
    qid_set.add(qid)
    label = int(line.split()[-1])
    if label == 0:
        if qid not in relevance_dict_0.keys():
            relevance_dict_0[qid] = [did]
        else:
            current_docs = relevance_dict_0.get(qid)
            current_docs.append(did)
            relevance_dict_0[qid] = current_docs
    if label == 1:
        if qid not in relevance_dict_1.keys():
            relevance_dict_1[qid] = [did]
        else:
            current_docs = relevance_dict_1.get(qid)
            current_docs.append(did)
            relevance_dict_1[qid] = current_docs
    if label == 2:
        if qid not in relevance_dict_2.keys():
            relevance_dict_2[qid] = [did]
        else:
            current_docs = relevance_dict_2.get(qid)
            current_docs.append(did)
            relevance_dict_2[qid] = current_docs
    if label == 3:
        if qid not in relevance_dict_3.keys():
            relevance_dict_3[qid] = [did]
        else:
            current_docs = relevance_dict_3.get(qid)
            current_docs.append(did)
            relevance_dict_3[qid] = current_docs
count_num = [0,0,0,0]
for keys_d in relevance_dict_0:
    count_num[0] = count_num[0] + len(relevance_dict_0.get(keys_d))
for keys_d in relevance_dict_1:
    count_num[1] = count_num[1] + len(relevance_dict_1.get(keys_d))
for keys_d in relevance_dict_2:
    count_num[2] = count_num[2] + len(relevance_dict_2.get(keys_d))
for keys_d in relevance_dict_3:
    count_num[3] = count_num[3] + len(relevance_dict_3.get(keys_d))
print(count_num)
#print(len(qid_set))

topn_bm25 = {}
topn_bert = {}

counter_list = [10, 100, 200, 500, 1000]
bm25_lines = open(bm25file, 'r').readlines()
bert_lines = open(bertfile, 'r').readlines()

index = int(len(bm25_lines)/1000)

for n in counter_list:
    #topn_bm25[n] = [0,0,0]
    #topn_bert[n] = [0,0,0]
    current_bm25 = [0, 0,0,0]
    current_bert = [0, 0,0,0]
    counter = 0
    for i in range(0,index):
        qid = bm25_lines[i*1000].split()[0]
        if qid not in qid_set:
            continue
        counter = counter+1
        count_bm25 = [0, 0, 0, 0]
        count_bert = [0, 0, 0, 0]
        for j in range(i*1000, i*1000+n):
            bm25_did = bm25_lines[j].split()[2]
            bert_did = bert_lines[j].split()[2]
            if relevance_dict_0.get(qid) is not None:
                #print(1)
                if bm25_did in relevance_dict_0.get(qid):
                    count_bm25[0] = count_bm25[0] + 1
                    #print(count_bm25)
                if bert_did in relevance_dict_0.get(qid):
                    count_bert[0] = count_bert[0] + 1

            if relevance_dict_1.get(qid) is not None:
                #print(1)
                if bm25_did in relevance_dict_1.get(qid):
                    count_bm25[1] = count_bm25[1] + 1
                    #print(count_bm25)
                if bert_did in relevance_dict_1.get(qid):
                    count_bert[1] = count_bert[1] + 1

            if relevance_dict_2.get(qid) is not None:
                #print(1)
                if bm25_did in relevance_dict_2.get(qid):
                    count_bm25[2] = count_bm25[2] + 1
                if bert_did in relevance_dict_2.get(qid):
                    count_bert[2] = count_bert[2] + 1

            if relevance_dict_3.get(qid) is not None:
                #print(1)
                if bm25_did in relevance_dict_3.get(qid):
                    count_bm25[3] = count_bm25[3] + 1
                if bert_did in relevance_dict_3.get(qid):
                    count_bert[3] = count_bert[3] + 1
        #print(count_bm25)
        if count_bm25[0] != 0:
            count_bm25[0] = count_bm25[0]/len(relevance_dict_0.get(qid))
        if count_bm25[1] != 0:
            count_bm25[1] = count_bm25[1]/len(relevance_dict_1.get(qid))
        if count_bm25[2] != 0:
            count_bm25[2] = count_bm25[2]/len(relevance_dict_2.get(qid))
        if count_bm25[3] != 0:
            count_bm25[3] = count_bm25[3]/len(relevance_dict_3.get(qid))

        if count_bert[0] != 0:
            count_bert[0] = count_bert[0]/len(relevance_dict_0.get(qid))
        if count_bert[1] != 0:
            count_bert[1] = count_bert[1]/len(relevance_dict_1.get(qid))
        if count_bert[2] != 0:
            count_bert[2] = count_bert[2]/len(relevance_dict_2.get(qid))
        if count_bert[3] != 0:
            count_bert[3] = count_bert[3]/len(relevance_dict_3.get(qid))

        #count_bm25 = [a/n for a in count_bm25]
        #count_bert = [a/n for a in count_bert]
        for k in range(0,4):
            current_bm25[k] = current_bm25[k]+ count_bm25[k]
            current_bert[k] = current_bert[k] + count_bert[k]
    #print(counter)
    current_bm25 = [a/counter for a in current_bm25]
    current_bert = [a/counter for a in current_bert]
    topn_bm25[n] = current_bm25
    topn_bert[n] = current_bert

out.write(str(count_num) + '\n')

for key in topn_bm25.keys():
    out.write("count " + str(key) + "\n")
    out.write("bm25_document: " + str(topn_bm25.get(key)) + "\n")
    out.write("bert_document: " + str(topn_bert.get(key)) + "\n")
    out.write("\n")









