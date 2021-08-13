import os
from tqdm import tqdm
base_dir = os.getcwd()+'/'
file_dir = base_dir + "test/" # input folder, modify this if want to test on other dataset

input_dir = file_dir + "data/"

output_file = open(file_dir + "bm25_original.res", 'w') # input bm25 file name, modify

file_input = input_dir + "pyserini"


output_file_dict = {}
qid_list = []

for i in tqdm(range(1,4)):
    file_lines = open(file_input+str(i)+ ".tsv", 'r').readlines()
    current_qid = ""
    for line in file_lines:
        items = line.split()
        qid = items[0]
        did = items[1]
        score = float(items[-1])
        if qid != current_qid:
            current_qid = qid
            if qid in qid_list:
                current_dict = output_file_dict.get(qid)
                current_dict[did] = score
                output_file_dict[qid] = current_dict
            else:
                qid_list.append(qid)
                current_dict = {}
                current_dict[did] = score
                output_file_dict[qid] = current_dict
        else:
            current_dict = output_file_dict.get(qid)
            current_dict[did] = score
            output_file_dict[qid] = current_dict

for q in tqdm(qid_list):
    current_dict = output_file_dict.get(q)
    sorted_dict =  {k: v for k, v in sorted(current_dict.items(), key=lambda item: item[1], reverse=True)}
    count = 1
    for key in sorted_dict.keys():
        output_file.write(q + '\t' + key + '\t' + str(sorted_dict.get(key)) + '\n')
        if count == 1000:
            break
        count = count+1
