# this file is used for getting the original bm25 and bert scores (not needed in the experiment)

#from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyserini.search import SimpleSearcher
import torch
import os
from tqdm import tqdm
print(SimpleSearcher.list_prebuilt_indexes())


#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')

searcher = SimpleSearcher('lucene-index-msmarco')
#model = AutoModelForSequenceClassification.from_pretrained("castorini/monobert-large-msmarco").eval()
#model = AutoModelForSequenceClassification.from_pretrained("castorini/monobert-large-msmarco").eval().cuda(device=0)
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if not os.path.exists("input"):
    os.mkdir("input")
input = open('dev.tsv', 'r')
check = open('dev.qrels.tsv', 'r')

checkLines = check.readlines()
qidset = set()

for line in checkLines:
    qidset.add(line.split()[0])

print(len(qidset))
print(qidset)


#bm25 = open('input/bm25.txt','w')
#bert = open('input/bert.txt','w')

Lines = input.readlines()
count = 0
for line in tqdm(Lines):
    id, query = line.split(maxsplit=1)
    if id not in qidset:
        continue
    hits = searcher.search(query, 1000)
    #max = hits[0].score
    #min = hits[len(hits)-1].score
    #dif = max-min
    bm25_lines = []
    bert_lines = []
    for i in range(len(hits)):
        #bm25.write(f'{id:15} {i + 1:4} {hits[i].docid:15} {(hits[i].score-min)/dif:.5f} \n' )
        bm25_lines.append(f'{id:15} {i + 1:4} {hits[i].docid:15} {(hits[i].score):.5f} \n')
        '''
        ret = tokenizer.encode_plus(query,
                                    searcher.doc(hits[i].docid).raw(),
                                    max_length=512,
                                    return_token_type_ids=True,
                                    return_tensors='pt', truncation=True)
        #input_ids = ret['input_ids']
        #tt_ids = ret['token_type_ids']
        input_ids = ret['input_ids'].to(DEVICE)
        tt_ids = ret['token_type_ids'].to(DEVICE)
        with torch.no_grad():
            #output, = model(input_ids, token_type_ids=tt_ids).to_tuple()
            output, = model(input_ids, token_type_ids=tt_ids)
            #print(output)
            reward = torch.nn.functional.softmax(output, 1)[:, -1]
            #print(reward.numpy()[0])
            #print(output)
            #print(reward.cpu().numpy())
            #bert_lines.append('{} {} {} \n'.format(id, hits[i].docid, reward.numpy()[0]))
            bert_lines.append('{} {} {} \n'.format(id, hits[i].docid, reward.cpu().numpy()[0]))
            #bert.write('{} {} {} \n'.format(id, hits[i].docid, reward.cpu().numpy()))
        '''
    with open("input/bm25_real.txt", "a+") as f:
        f.writelines(bm25_lines)
    '''
    with open("input/bert.txt", "a+") as f:
        f.writelines(bert_lines)
    '''
    del bm25_lines, bert_lines
    #count = count+1
    #if count is 2:
        #break










