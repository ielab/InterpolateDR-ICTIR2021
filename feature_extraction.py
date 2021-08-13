import os
import glob
from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
import torch
from tqdm import tqdm
import math
import statistics


searcher = SimpleSearcher('lucene-index-msmarco')
index_reader = IndexReader('lucene-index-msmarco')
statics = index_reader.stats()
N = statics['documents']
tokencoll = statics['unique_terms']
tokencoll = 2660824
#print(tokencoll)
#allset = {}
#allset["mrr3"]  = glob.glob("eval/mrr3/*.eval")
#allset["ndcg"]  = glob.glob("eval/ndcg/*.eval")
#allset["ndcg10"]  = glob.glob("eval/ndcg10/*.eval")
#allset["ndcg20"] = glob.glob("eval/ndcg20/*.eval")

base_dir = os.getcwd()+'/'
quriesfile_o = base_dir + "Msmarco/qrels/" + "dev.tsv"
quriesfile_e = base_dir + "Msmarco/qrels/" + "dev.qrels.tsv" #queries folder location, modify this if want to test on other dataset
file_dir = base_dir + "Msmarco/ance/" # input folder, modify this if want to test on other dataset

#mrr10 = glob.glob(file_dir + "eval/mrr10/*.eval")
mrr10 = glob.glob(file_dir + "eval/mrr10/*.eval")

if not os.path.exists(file_dir+"feature_out"):
    os.mkdir(file_dir+"feature_out")

input_dev = open(quriesfile_o, 'r')
input_bm = open(file_dir+"bm25.txt", 'r')
input_be = open(file_dir+"bert.txt", 'r')
output = open(file_dir+'feature_out/features.txt', 'w')
bm25_set = {}
bert_set = {}
bm25 = input_bm.readlines()
bert = input_be.readlines()

mrr10_list = {}
mrr10_key = {}
for element in tqdm(range(0,11)):
    k = element/10
    #k = k.replace('.eval', '')
    file = file_dir + 'eval/mrr10/' + str(k) + '.eval'
    current_eval = open(file, 'r')
    Eval_lines = current_eval.readlines()
    for eval_line in Eval_lines:
        if (eval_line.split()[1] not in mrr10_list.keys()) or (
                float(eval_line.split()[-1]) > mrr10_list.get(eval_line.split()[1])):
            mrr10_key[eval_line.split()[1]] = [int(float(k)*10)]
            mrr10_list[eval_line.split()[1]] = float(eval_line.split()[-1])
        elif float(eval_line.split()[-1]) == mrr10_list.get(eval_line.split()[1]):
            current_list = mrr10_key.get(eval_line.split()[1])
            current_list.append(int(float(k)*10))
            mrr10_key[eval_line.split()[1]] = current_list
print(mrr10_list)
for l in tqdm(bm25):
    qid = l.split()[0]
    did = l.split()[-2]
    score = float(l.split()[-1])
    if qid in bm25_set.keys():
        score_set = list(bm25_set.get(qid))
        score_set.append(score)
        bm25_set[qid] = score_set
    else:
        score_set = []
        score_set.append(score)
        bm25_set[qid] = score_set

for l in tqdm(bert):
    qid = l.split()[0]
    did = l.split()[-2]
    score = float(l.split()[-1])
    if qid in bert_set.keys():
        score_set = list(bert_set.get(qid))
        score_set.append(score)
        bert_set[qid] = score_set
    else:
        score_set = []
        score_set.append(score)
        bert_set[qid] = score_set

for bert_id in tqdm(bert_set.keys()):
    value_set = bert_set.get(bert_id)
    value_set = sorted(value_set, key=float, reverse=True)
    bert_set[bert_id] = value_set

Lines = input_dev.readlines()
for line in tqdm(Lines):
    id, query = line.split(maxsplit=1)
    if id in mrr10_key.keys():
        query_terms = query.split()
        ql = len(query_terms) #query length def 1
        idft = []
        #document_set = set()
        #document_set = ['hgtygrtgrt']
        SCS = 0
        multiplied = 1
        if (id in bm25_set.keys()) and (id in bert_set.keys()):
            bm25_mean = sum(bm25_set.get(id))/len(bm25_set.get(id))
            bert_mean = sum(bert_set.get(id))/len(bert_set.get(id))
            bm25_mean_100 = sum(bm25_set.get(id)[0:99])/100
            bert_mean_100 = sum(bert_set.get(id)[0:99]) / 100
            bm25_mean_10 = sum(bm25_set.get(id)[0:9])/10
            bert_mean_10 = sum(bert_set.get(id)[0:9]) / 10
            bm25_sd = statistics.stdev(bm25_set.get(id))
            bert_sd = statistics.stdev(bert_set.get(id))
            bm25_sd_100 = statistics.stdev(bm25_set.get(id)[0:99])
            bert_sd_100 = statistics.stdev(bert_set.get(id)[0:99])
            bm25_sd_10 = statistics.stdev(bm25_set.get(id)[0:9])
            bert_sd_10 = statistics.stdev(bert_set.get(id)[0:9])
            bm25_max = max(bm25_set.get(id))
            bert_max = max(bert_set.get(id))

            for term in query_terms:
                analyzed = index_reader.analyze(term)
                if len(analyzed) != 0:
                    qtf = query_terms.count(term)
                    df,cf = index_reader.get_term_counts(analyzed[0], analyzer=None)
                    pml = qtf/ql
                    pcoll = cf/tokencoll
                    #if pml > 0 and pcoll > 0:
                    if pcoll != 0 and pml != 0:
                        SCS = SCS + (pml* math.log(2, pml/pcoll)) # SCS value def 5
                        multiplied = multiplied*(tokencoll/cf)
                    #postings_list = index_reader.get_postings_list(analyzed[0], analyzer=None)
                    #if postings_list is not None:
                        #for posting in postings_list:
                            #a=0
                            #document_set.add(posting.docid)
                    if df != 0:
                        idft.append(math.log(2, N + 0.5) / df / (math.log(2, N + 1)))
            y1 = 0

            if len(idft) != (1 or 0):
                y1 = statistics.stdev(idft) # y1 value def 2
            y2 = max(idft)/min(idft) # y2 value def 3
            #w = -math.log(len(document_set)/N) # w value def 4
            if multiplied != 0:
                AvICTF = math.log(2,multiplied)/ql
            with open(file_dir+ "feature_out/features.txt", "a+") as f:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(id, ql, y1, y2, SCS, AvICTF, bm25_mean, bm25_mean_100, bm25_mean_10, bm25_max, bert_mean, bert_mean_100, bert_mean_10, bert_max, bm25_sd, bm25_sd_100, bm25_sd_10, bert_sd, bert_sd_100, bert_sd_10))
                for i in range(0,11):
                    if i<min(mrr10_key.get(id)) or i>max(mrr10_key.get(id)):
                        f.write('0\t')
                    else:
                        f.write('1\t')
                f.write('\n')











