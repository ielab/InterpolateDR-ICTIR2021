#this is the run file for intepolation, it takes the initial run file from bm25 and DR as in the paper, then it will
#normalise and intepolate their scores, then doing evaluation using various metrics, finally generate graphs.

import os
base_dir = os.getcwd()+'/'
year = 2020 # if getting results on msmarco, setting year to 2019
quriesfile = base_dir + "Deepdl/2020_qrels/" + "dev.qrels.tsv" #queries folder location, modify this if want to test on other dataset
file_dir = base_dir + "Deepdl/Repbert_BM25/2020/" # input folder, modify this if want to test on other dataset
graph_file_dir = base_dir+"all_graphs/DeepDL/rep/2020/" # graph folder, modify the location of the output graph.
input_bm25 = "pyserini_DL2020_bm25_top2000.tsv" # input bm25 file name, modify
input_bert = "RepBERT_DL2020_top2000.res" # input bert_based DR file name, modify

#above are parameters needed to set

input_file_dir = file_dir+"input/"
eval_file_dir = file_dir+"eval/"
ouput_file_dir = file_dir+"output/"

os.system("source ~/.bashrc activate")
if not os.path.exists(ouput_file_dir):
    if (not os.path.exists(file_dir+"bm25.txt")) or (not os.path.exists(file_dir+"bert.txt")):
        os.system("python3 input_normalise.py " + file_dir + " " + input_bm25 + " " + input_bert)
    os.system("python3 combine_score_2000.py " + file_dir)

eval_dict = {}
if year==2020:
    eval_dict["recall.10 -l 2"] = eval_file_dir + "recall10/"
    eval_dict["recall.200 -l 2"] = eval_file_dir + "recall200/"
    eval_dict["recall.500 -l 2"] = eval_file_dir + "recall500/"
    eval_dict["recall.1000 -l 2"] = eval_file_dir + "recall/"
    eval_dict["map.1000 -l 2"] = eval_file_dir + "map/"
else:
    eval_dict["recall.10"] = eval_file_dir+"recall10/"
    eval_dict["recall.200"] = eval_file_dir+"recall200/"
    eval_dict["recall.500"] = eval_file_dir+"recall500/"
    eval_dict["recall.1000"] = eval_file_dir+"recall/"
    eval_dict["map"] = eval_file_dir + "map/"

eval_dict["recip_rank -M 10"] = eval_file_dir+"mrr10/"

eval_dict["ndcg_cut.10"] = eval_file_dir+"ndcg10/"
eval_dict["ndcg_cut.200"] = eval_file_dir+"ndcg200/"
eval_dict["ndcg_cut.500"] = eval_file_dir+"ndcg500/"
eval_dict["ndcg_cut.1000"] = eval_file_dir+"ndcg1000/"


for k, v in eval_dict.items():
    if not os.path.exists(v):
        os.mkdir(v)
        for i in range(0,11):
            out_value = str(i/10)
            os.system(base_dir+"/trec_eval/trec_eval" + " -q -m " + k + " " + quriesfile + " " + ouput_file_dir + "output" + out_value+".txt" + " > " + v + out_value +".eval")

os.system("python3 eval_graph.py " + file_dir + " " + graph_file_dir)

