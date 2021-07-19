# the use of this file is mainly for conducting the t-test in the experienment.
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import chain
from scipy import stats
from multipy.data import neuhaus
from multipy.fwer import bonferroni

"""
if not os.path.exists("graph"):
    os.mkdir("graph")

if not os.path.exists("graph/general"):
    os.mkdir("graph/general")
"""
folder = "/Volumes/Samsung_T5/2020_summer/BERT-summer-project/"
#graph_output = "/Volumes/Samsung_T5/2020_summer/BERT-summer-project/all_graphs/"

#input_folder = folder + "Deepdl/ANCE_BM25_dl/2000/2020/"
#input_folder2 = folder + "Deepdl/Ret_Bert_reranker/2000/2020/"
input_folder = folder + "Msmarco/2000/ance/"
input_folder2 = folder + "Msmarco/2000/rep/"
evalu = {}

allset = {}
allset2 = {}

allset["mrr10"] = glob.glob(input_folder + "eval/mrr10/*.eval")
#allset["mrr3"]  = glob.glob("eval/mrr3/*.eval")
#allset["ndcg"]  = glob.glob("eval/ndcg/*.eval")

allset["ndcg10"]  = glob.glob(input_folder+"eval/ndcg10/*.eval")
allset["map"] = glob.glob(input_folder + "eval/map/*.eval")
allset["recall"] = glob.glob(input_folder + "eval/recall/*.eval")
allset["ndcg1000"] = glob.glob(input_folder + "eval/ndcg1000/*.eval")


allset2["ndcg10"]  = glob.glob(input_folder2+"eval/ndcg10/*.eval")
allset2["map"] = glob.glob(input_folder2 + "eval/map/*.eval")
allset2["recall"] = glob.glob(input_folder2 + "eval/recall/*.eval")
allset2["ndcg1000"] = glob.glob(input_folder2 + "eval/ndcg1000/*.eval")
allset2["mrr10"] = glob.glob(input_folder2 + "eval/mrr10/*.eval")
#allset["ndcg20"] = glob.glob(input_folder + "eval/ndcg20/*.eval")

for key in allset:
    one_and_0 = 0
    score_set = []
    Best_list = {}
    Best_key = {}
    all_value_set = []
    significant_list = []
    for element in allset.get(key):
        k = element.split('/')[-1]
        k = k.replace('.eval', '')
        current_eval = open(element, 'r')
        Lines = current_eval.readlines()

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

    list2 = {}
    list5 = {}
    list_best = {}

    best_score = sum(Best_list.values())/len(Best_list.values())

    for element in allset.get(key):
        k = element.split('/')[-1]
        k = k.replace('.eval', '')
        current_eval = open(element, 'r')
        Lines = current_eval.readlines()
        for line_index in range(0, len(Lines)):
            line2 = Lines[line_index]

            items2 = line2.split()
            if k=="0.0":
                list2[items2[1]] = float(items2[-1])

            if k=="0.5":
                list5[items2[1]] = float(items2[-1])


        #print(input_folder + key + " with alpha " + k  + " = " + str(p) + " with best score = " + str(best_score) + " and k score = " + str(sum(l2)/len(l2)) )
    l0 = []
    lbest = []
    l5 = []

    for key4 in Best_list.keys():
        lbest.append(Best_list.get(key4))
        l0.append(list2.get(key4))
        l5.append(list5.get(key4))
    _, p1 = stats.ttest_ind(l0, lbest)
    _, p2 = stats.ttest_ind(l0, l5)
    evalu[key] = [p1, p2]
    print("key:" + key + " p1: " + str(p1) + " " + "p2: " + str(p2) + " best_score: " + str(best_score))




print("\n")

for key in allset2:

    one_and_0 = 0
    score_set = []
    Best_list = {}
    Best_key = {}
    all_value_set = []
    significant_list = []

    for element in allset2.get(key):
        k = element.split('/')[-1]
        k = k.replace('.eval', '')
        current_eval = open(element, 'r')
        Lines = current_eval.readlines()

        for line in Lines:
            if (line.split()[1] not in Best_list) or (float(line.split()[-1]) > Best_list.get(line.split()[1])):
                # print(str(Best_list.get(line.split()[1])) + "  " + line.split()[-1] + "with number " + k + " " + line.split()[1])
                Best_key[line.split()[1]] = [float(k)]
                Best_list[line.split()[1]] = float(line.split()[-1])
            elif float(line.split()[-1]) == Best_list.get(line.split()[1]):
                current_list = Best_key.get(line.split()[1])
                current_list.append(float(k))
                Best_key[line.split()[1]] = current_list
        averagescore_line = Lines[-1]
        averagescore = float(averagescore_line.split()[-1])
        score_set.append([float(k), averagescore])

    list2 = {}
    list5 = {}
    list_best = {}

    best_score = sum(Best_list.values()) / len(Best_list.values())

    for element in allset2.get(key):
        k = element.split('/')[-1]
        k = k.replace('.eval', '')
        current_eval = open(element, 'r')
        Lines = current_eval.readlines()
        for line_index in range(0, len(Lines)):
            line2 = Lines[line_index]

            items2 = line2.split()
            if k == "0.0":
                list2[items2[1]] = float(items2[-1])

            if k == "0.5":
                list5[items2[1]] = float(items2[-1])

        # print(input_folder + key + " with alpha " + k  + " = " + str(p) + " with best score = " + str(best_score) + " and k score = " + str(sum(l2)/len(l2)) )

    l0 = []
    lbest = []
    l5 = []

    for key4 in Best_list.keys():
        lbest.append(Best_list.get(key4))
        l0.append(list2.get(key4))
        l5.append(list5.get(key4))

    _, p1 = stats.ttest_ind(l0, lbest)
    _, p2 = stats.ttest_ind(l0, l5)
    lll = evalu.get(key)
    lll.append(p1)
    lll.append(p2)
    evalu[key] = lll

    print("key:" + key + " p1: " + str(p1) + " " + "p2: " + str(p2) + " best_score: " + str(best_score))

#print(evalu)
for key6 in evalu:
    current_l = evalu.get(key6)
    significant_pvals = bonferroni(current_l, alpha=0.05)
    print(key6)
    print(['{:.4f}'.format(p) for p in current_l], significant_pvals)


