# this file will normalise the score obtained by original input and normalise them to get normalised scores,
# the file will need three inputs, the first is the entry folder name, the second is the bm25 original score file name,
# the third input is the DR original scores file name
from tqdm import  tqdm
import sys
import os

base_dir = os.getcwd() + '/'
entry_file = base_dir+ "Msmarco/ance/" #modify for further changes

if len(sys.argv)>=2:
    entry_file = sys.argv[1]

if True:
    #input = open(entry_file+ "input/pyserini_msmarco_bm25_top2000.tsv", 'r')
    if len(sys.argv) >=2:
        input = open(entry_file + 'input/' + sys.argv[2], 'r')
    output = open(entry_file + "bm25.txt", 'w')
    lines = input.readlines()
    previous = lines[0].split()[0]
    score = 0.0
    score_list = []
    #print(len(lines))
    #print(len(input2.readlines()))
    new_line_list = []
    for line_index, line in tqdm(enumerate(lines)):
        items = line.split()
        qid = items[0]
        #rank = items[3]
        did = items[2]
        score = float(items[-2])
        new_line = str(qid) + ' ' + did + ' '
        if line_index == len(lines)-1:
            score_list.append(score)
            new_line_list.append(new_line)
        if qid != previous or line_index==len(lines)-1:
            previous = qid
            diff = max(score_list) - min(score_list)
            mini = min(score_list)
            for index in range(0, len(score_list)):
                norm_score = 1
                if diff!=0:
                    norm_score = (score_list[index]-mini)/diff
                norm_line = new_line_list[index] + str(norm_score) + '\n'
                output.write(norm_line)
            score_list.clear()
            new_line_list.clear()

        score_list.append(score)
        new_line_list.append(new_line)
    lines.clear()
    output.close()

if True:
    #input = open(entry_file+ "input/ANCE_msmarco_top2000.res", 'r')
    if len(sys.argv) >=2:
        input = open(entry_file+  'input/' + sys.argv[3], 'r')
    output = open(entry_file+"bert.txt", 'w')
    lines = input.readlines()
    previous = lines[0].split()[0]
    score = 0.0
    score_list = []
    new_line_list = []
    for line_index, line in tqdm(enumerate(lines)):
        items = line.split()
        qid = items[0]
        # rank = items[3]
        did = items[2]
        score = float(items[-2])
        new_line = str(qid)  + ' ' + did + ' '

        if line_index == (len(lines)-1):
            score_list.append(score)
            new_line_list.append(new_line)

        if qid != previous or (line_index==(len(lines)-1)):
            previous = qid
            diff = max(score_list) - min(score_list)
            mini = min(score_list)
            for index in range(0, len(score_list)):
                norm_score = (score_list[index] - mini) / diff
                norm_line = new_line_list[index] + str(norm_score) + '\n'
                output.write(norm_line)
            score_list.clear()
            new_line_list.clear()
        score_list.append(score)
        new_line_list.append(new_line)
    lines.clear()
    output.close()



