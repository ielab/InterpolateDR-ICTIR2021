#this is the run file for intepolation, it takes the initial run file from bm25 and DR as in the paper, then it will
#normalise and intepolate their scores, then doing evaluation using various metrics, finally generate graphs.

import os
base_dir = os.getcwd()+'/'
year = 2019 # if getting results on msmarco, setting year to 2019

quriesfile2 = base_dir + "Deepdl/2019_qrels/" + "dev.qrels.tsv" #queries folder location, modify this if want to test on other dataset
quriesfile1 = base_dir + "Deepdl/2019_qrels/" + "dev.qrels.tsv" #queries folder location, modify this if want to test on other dataset
file_dir = base_dir + "Deepdl/adore_tilde/2019/" # input folder, modify this if want to test on other dataset
graph_file_dir = base_dir+"all_graphs/adore-tilde/2019/" # graph folder, modify the location of the output graph.
input_bm25 = "run.msmarco-passage-unicoil-TILDE-200-dl2019.trec" # input bm25 file name, modify
input_bert = "testout.rank.tsv" # input bert_based DR file name, modify

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
    #eval_dict["recall.10"] = eval_file_dir+"recall10/"
    eval_dict["recall.100"] = eval_file_dir+"recall100/"
    #eval_dict["recall.500"] = eval_file_dir+"recall500/"
    eval_dict["recall.1000"] = eval_file_dir+"recall/"
    eval_dict["map"] = eval_file_dir + "map/"

#eval_dict["recip_rank -M 10"] = eval_file_dir+"mrr10/"
eval_dict["recip_rank"] = eval_file_dir + "rr/"
eval_dict["ndcg_cut.10"] = eval_file_dir+"ndcg10/"
#eval_dict["ndcg_cut.200"] = eval_file_dir+"ndcg200/"
#eval_dict["ndcg_cut.500"] = eval_file_dir+"ndcg500/"
#eval_dict["ndcg_cut.1000"] = eval_file_dir+"ndcg1000/"


for k, v in eval_dict.items():
    if not os.path.exists(v):
        os.mkdir(v)
        if k == "recip_rank -M 10" or k== "recip_rank":
            for i in range(0, 11):
                out_value = str(i / 10)
                os.system(
                    base_dir + "/trec_eval/trec_eval" + " -q -m " + k + " " + quriesfile2 + " " + ouput_file_dir + "output" + out_value + ".txt" + " > " + v + out_value + ".eval")
        else:
            for i in range(0,11):
                out_value = str(i/10)
                os.system(base_dir+"/trec_eval/trec_eval" + " -q -m " + k + " " + quriesfile2 + " " + ouput_file_dir + "output" + out_value+".txt" + " > " + v + out_value +".eval")

#os.system("python3 eval_graph.py " + file_dir + " " + graph_file_dir)

#trec_eval -m recip_rank Deepdl/2019_qrels/dev.qrels.tsv Deepdl/UniTIL_bm25/2019/128/output/output0.0.txt