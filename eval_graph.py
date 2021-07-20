# this file have two optionsm one is to using bash input which need to have the first input contain the input folder location
# and the second input as output folder location (embedded in run.py), another way is to modify the code below in entry_folder,
# input_folder, output_folder to run different runs.

import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import chain
from scipy import stats
import sys


#folder = "/Volumes/Samsung_T5/2020_summer/BERT-summer-project/"
#graph_output = "/Volumes/Samsung_T5/2020_summer/BERT-summer-project/all_graphs/"
entry_folder = os.getcwd()+'/'
input_folder = entry_folder+ "Deepdl/ANCE_BM25/2020/"
output_folder = entry_folder + "all_graphs/DeepDL/ance/2020/"

if len(sys.argv)>=1:
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

histout_folder = output_folder+'histogram/'
if not os.path.exists(histout_folder):
    os.mkdir(histout_folder)

allset = {}
allset["mrr10"] = glob.glob(input_folder + "eval/mrr10/*.eval")
#allset["mrr3"]  = glob.glob("eval/mrr3/*.eval")
#allset["ndcg"]  = glob.glob("eval/ndcg/*.eval")
allset["ndcg10"]  = glob.glob(input_folder+"eval/ndcg10/*.eval")
allset["ndcg200"]  = glob.glob(input_folder+"eval/ndcg200/*.eval")
allset["ndcg500"]  = glob.glob(input_folder+"eval/ndcg500/*.eval")
allset["ndcg1000"]  = glob.glob(input_folder+"eval/ndcg1000/*.eval")
allset["map"] = glob.glob(input_folder + "eval/map/*.eval")
allset["recall"] = glob.glob(input_folder + "eval/recall/*.eval")
allset["recall10"] = glob.glob(input_folder + "eval/recall10/*.eval")
allset["recall200"] = glob.glob(input_folder + "eval/recall200/*.eval")
allset["recall500"] = glob.glob(input_folder + "eval/recall500/*.eval")
#allset["ndcg20"] = glob.glob(input_folder + "eval/ndcg20/*.eval")

for key in allset:
    #if not os.path.exists("graph/"+key):
        #os.mkdir("graph/"+key)
    one_and_0 = 0
    output = output_folder + key + ".png"
    histout = histout_folder + key + ".png"
    score_set = []
    Best_list = {}
    Best_key = {}
    all_value_set = []
    input1 = open(input_folder + "eval/"+ key +"/0.0.eval", 'r')
    lines1 = input1.readlines()
    significant_list = []
    a0_list = []
    for element in allset.get(key):
        #a0_list = []
        k = element.split('/')[-1]
        k = k.replace('.eval', '')
        #print(k)
        current_eval = open(element, 'r')
        Lines = current_eval.readlines()
        list1 = []
        list2 = []
        for line_index in range(0, len(lines1)):
            line1 = lines1[line_index]
            line2 = Lines[line_index]
            items1 = line1.split()
            items2 = line2.split()
            list1.append(float(items1[-1]))
            list2.append(float(items2[-1]))
        a0_list = list1
        _, p = stats.ttest_ind(list1, list2)


        #print(p)
        if p < 0.05:
            significant_list.append(float(k))
        for line in Lines:
            if (line.split()[1] not in Best_list) or (float(line.split()[-1]) > Best_list.get(line.split()[1])):
                #print(str(Best_list.get(line.split()[1])) + "  " + line.split()[-1] + "with number " + k + " " + line.split()[1])
                Best_key[line.split()[1]] = [float(k)]
                Best_list[line.split()[1]] = float(line.split()[-1])
            elif float(line.split()[-1]) == Best_list.get(line.split()[1]):
                current_list = Best_key.get(line.split()[1])
                current_list.append(float(k))
                Best_key[line.split()[1]] = current_list
        averagescore_line = Lines[-1]
        averagescore = float(averagescore_line.split()[-1])
        score_set.append([float(k), averagescore])
    #print(Best_list)
    bestvalue = sum(Best_list.values())/len(Best_list.values())
    oracle_list = list(Best_list.values())
    value_set = Best_key.values()
    #print(Best_list.values())
    _, p = stats.ttest_ind(a0_list, oracle_list)
    #print(Best_key.keys())

    flatten_list = list(chain.from_iterable(value_set))
    listwithcoun={}
    for index in range(0,11):
        current_index = index/10
        occurance = flatten_list.count(current_index)
        listwithcoun[str(current_index)] = occurance
    for v in Best_key.keys():
        c = Best_key.get(v)
        if (0.0 in c):
            one_and_0 +=1
    #print(one_and_0)
    plt.bar(listwithcoun.keys(), listwithcoun.values(), 1.0)
    plt.tight_layout()
    plt.savefig(histout)
    plt.close()
    score_set = sorted(score_set)
    keylist = [i[0] for i in score_set]
    valuelist = [i[1] for i in score_set]
    x = np.linspace(0, 1)
    y = bestvalue + 0*x
    plt.xticks(keylist)
    plt.xlabel('α', fontsize = 20)
    plt.ylabel( key, fontsize = 20)
    normal, =plt.plot(keylist,valuelist)
    print(valuelist)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)

    significant_value_list = []
    significant_list = sorted(significant_list)
    for significant_point in significant_list:
        point_index = int(float(significant_point)*10)
        significant_value_list.append(valuelist[point_index])

    plt.scatter(significant_list, significant_value_list, marker="*")
    best, = plt.plot(x, y, '-r', label='bestvalue')
    if p<0.05:
        plt.scatter([0], bestvalue, marker="*", edgecolors='red')
    print(output_folder + key + " = " + str(bestvalue))
    plt.legend([best,normal ], ["oracle score", "interpolation"], prop={"size":20})
    plt.tight_layout()
    plt.savefig(output)
    plt.close()



