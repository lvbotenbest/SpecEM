
import bert_score
import json
import os
from typing import List, Optional, Union, Dict, Tuple
import torch 
from .huggingface_bleu import Bleu
from rouge import Rouge
import jieba
import sys
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

sys.setrecursionlimit(1000000)


device = torch.device("cuda")



def eval_bertscore(
    hypotheses: List[List[str]],
    references: List[List[str]],
    model_type="bert-base-multilingual-cased",
    lang="en",
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using bertscore.
    BertScore officially recommends using microsoft/deberta-xlarge-mnli as the model.
    the default multilingual model is bert-base-multilingual-cased.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    logger.info("Evaluating bertscore")
    assert len(hypotheses) == len(references)
    if lang=="en":
        P, R, F1 = bert_score.score(hypotheses, references, lang=lang, verbose=True, model_type="model path",num_layers=9, batch_size=128)
    else:
        P, R, F1 = bert_score.score(hypotheses, references, lang=lang, verbose=True, model_type="model path",num_layers=10, batch_size=128)
   
 
   
    return P.mean(),R.mean(),F1.mean()




def calculate_score(hypotheses,references):

    logger.info(str(len(hypotheses)))

    metric={}



    P,R,F1 = eval_bertscore(hypotheses, references,lang="en")


    metric["bertscore P"]=round(P.item()*100, 4)
    metric["bertscore R"]=round(R.item()*100, 4)
    metric["bertscore F"]=round(F1.item()*100, 4)


    bleu_reference=[]
    for i in references:
        bleu_reference.append([i])

    bleu = Bleu()
    bleuscores=bleu.compute(predictions=hypotheses,references=bleu_reference)

    metric["scarebleu"]=round(bleuscores["bleu"]*100, 4)
    metric["bleu-1"]=round(bleuscores["precisions"][0]*100, 4)
    metric["bleu-2"]=round(bleuscores["precisions"][1]*100, 4)
    metric["bleu-3"]=round(bleuscores["precisions"][2]*100, 4)
    metric["bleu-4"]=round(bleuscores["precisions"][3]*100, 4)

    # print(metric)
    rouge = Rouge()

    rougescores=rouge.get_scores(hypotheses, references, avg = True)

    metric["rouge-1"]=round(rougescores["rouge-1"]["f"]*100, 4)
    metric["rouge-2"]=round(rougescores["rouge-2"]["f"]*100, 4)
    metric["rouge-l"]=round(rougescores["rouge-l"]["f"]*100, 4)

    logger.info(str(metric))



def calculate_score_cn(hypotheses,references,use_jieba=False):

    logger.info(str(len(hypotheses)))

    metric={}


    P,R,F1 = eval_bertscore(hypotheses, references,lang="zh")


    metric["bertscore P"]=round(P.item()*100, 4)
    metric["bertscore R"]=round(R.item()*100, 4)
    metric["bertscore F"]=round(F1.item()*100, 4)


    bleu_reference=[]

    
    if use_jieba:
        bleu_reference = [[" ".join(jieba.cut(i))] for i in references]
        hypotheses_data = [ " ".join(jieba.cut(i)) for i in hypotheses]
        references_data = [ " ".join(jieba.cut(i)) for i in references]
    else:
        bleu_reference = [[" ".join(i)] for i in references]
        hypotheses_data = [ " ".join(i) for i in hypotheses]
        references_data = [ " ".join(i) for i in references]


    bleu = Bleu()
    bleuscores=bleu.compute(predictions=hypotheses_data,references=bleu_reference)

    metric["scarebleu"]=round(bleuscores["bleu"]*100, 4)
    metric["bleu-1"]=round(bleuscores["precisions"][0]*100, 4)
    metric["bleu-2"]=round(bleuscores["precisions"][1]*100, 4)
    metric["bleu-3"]=round(bleuscores["precisions"][2]*100, 4)
    metric["bleu-4"]=round(bleuscores["precisions"][3]*100, 4)

    

    rouge = Rouge()

    rougescores=rouge.get_scores(hypotheses_data, references_data, avg=True)

    metric["rouge-1"]=round(rougescores["rouge-1"]["f"]*100, 4)
    metric["rouge-2"]=round(rougescores["rouge-2"]["f"]*100, 4)
    metric["rouge-l"]=round(rougescores["rouge-l"]["f"]*100, 4)

    logger.info(str(metric))


def read_file(input_file):
    with open(input_file, 'r') as f:
  
        data = json.load(f)

    hypotheses=[]
    references=[]
    for i in data:
        hypotheses.append(i["predict_answer"])
        references.append(i["truth_answer"])
    return hypotheses,references

if __name__ == '__main__':


    file3 = "file path"
    hypotheses3,references3 = read_file(file3)
    calculate_score(hypotheses3,references3)


