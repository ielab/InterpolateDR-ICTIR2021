from tqdm import  tqdm
import sys
entry_file = sys.argv[1]
if True:
    input = open(entry_file + sys.argv[2], 'r')
    output = open(entry_file + "bm25.txt", 'w')
    lines = input.readlines()
    previous = lines[0].split()[0]
    score = 0.0
    score_list = []
    #print(len(lines))
    #print(len(input2.readlines()))
    new_line_list = []
    for line in tqdm(lines):
        items = line.split()
        qid = items[0]
        #rank = items[3]
        did = items[2]
        score = float(items[-2])
        new_line = str(qid) + ' ' + did + ' '

        if qid != previous:
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

if True:
    input = open(entry_file+sys.argv[3], 'r')
    output = open(entry_file+"bert.txt", 'w')
    lines = input.readlines()
    previous = lines[0].split()[0]
    score = 0.0
    score_list = []
    # print(len(lines))
    # print(len(input2.readlines()))
    new_line_list = []
    for line in tqdm(lines):
        items = line.split()
        qid = items[0]
        # rank = items[3]
        did = items[1]
        score = float(items[-1])
        new_line = str(qid)  + ' ' + did + ' '

        if qid != previous:
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



