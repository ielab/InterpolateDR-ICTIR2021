# InterpolateDR-ICTIR2021
### BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval
***
#### Environment setting:
Please download all corresponding library listed in requirements.txt for fully utilization of the code.
<br/>*Note that the project is based on python3, python2 may not be fully ultilisable*
***
#### Code Explanation
* **run.py:** This file will give a full work through of the project intepolation. By feeding the input folder location and output graph folder location, this file will firstly normalise the original bm25 and dr scores, then combine them as paper stated, then linearly interpolate the scores to create output. Next, the scores are evaluated and graphs will be generated in the stated output folder using the eval file.
* **input_normalise.py:** This python file will input the original bm25 scores and DR scores and output the normalised scores in the corresponding entry folder input folder.
* **combine_score_2000.py:** This python file will input the normalised bm25 scores and DR scores and linearly interpolate them using parameter alpha.
* **eval_graph.py:** This python file will ultilize the eval files generated and create plots based on these eval files.
* **get_cuttoff_relevance.py:** This file is used to get the cutoff relevance of each relevant levels as stated in the paper.
* **t_test.py:** This file is used for conducting t_test in the experiment, as stated in the paper
* **unjudged_count.py:** This file is used to count the unjudged documents for each dataset. 

***
#### Folder Description
* **Msmarco:** includes all input folders for ms marco dataset
* **Deepdl:** includes all input folders for deep dl dataset (including 2019 and 2020)
* **all_graphs:** includes all graphs generated and used in the experiment

***
#### Input Generation
* **ANCE:** Generation of ance scores can refer to [ANCE](https://github.com/microsoft/ANCE/)
* **REP_bert:** Generation of Repbert scores can refer to [Repbert](https://github.com/jingtaozhan/RepBERT-Index)
* **BM25:** Generation of BM25 scores can refer to [BM25](https://github.com/castorini/pyserini)

***
#### Workflow:
* **1:** Put files you want to do interpolation inside a directory/input/
* **2:** Change parameters in run.py (see comments) to define input folder, output folder, input run names, and qrel file location.
* **3:** Use python3 run.py to get interpolation (it will do normalisation first, then interpolate the normlised scores, and do evaluations and then draw graphs)

