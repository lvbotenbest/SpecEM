import torch
import concurrent.futures
import time
import torch.nn.functional as F
import numpy as np
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class EnsembleService(object):
    def __init__(self, inference_model_lists, verify_model_lists,max_generate_tokens, each_step_max_generate_token,learn_rate=False ):  

        self.inference_model_lists = inference_model_lists
        self.verify_model_lists = verify_model_lists

        self.max_generate_tokens = max_generate_tokens
        self.each_step_max_generate_token = each_step_max_generate_token
      
        self.org_model_list = [model.model_type  for model in self.inference_model_lists]
        self.model_id_dic = {self.org_model_list[i]: i for i in range(len(self.org_model_list))}

        self.K = len(self.verify_model_lists)
        self.eta = 0

        self.w = np.ones(self.K) / self.K
        
        if learn_rate == 1000:
            self.Use_Hedge = False
        else:
            self.Use_Hedge = True
            self.learn_rate = float(learn_rate/10)
            

    def generate(self,input_text,short_decoder=True,Output_stream=False):
        max_tokens = self.max_generate_tokens
        eos_flag = True
        best_response = ""
        best_new_generation = input_text
        Fast_decoder = short_decoder
        
        model_list = [model.model_type  for model in self.inference_model_lists]

        model_container = [ i for i in self.inference_model_lists]
        verify_model_container = [ i for i in self.verify_model_lists]

        model_hit_times = {}
        for key in model_list:
            model_hit_times[key]=0

        self.Clear_Cache()

        turns_num = 0
        call_model_nums = 0

        hit_list=[]

        
        while eos_flag and len(best_response.split())<max_tokens and self.cn_str_count(best_response)<max_tokens:    
            turns_num += 1
            call_model_nums += len(model_list)

            new_generations,stop_space = self.Parallel_Inference(best_new_generation,model_container)
                       
            if Fast_decoder:
                eos_model = self.find_if_has_eos_value(stop_space)
                if eos_model is not None:
                    best_new_generation = new_generations[eos_model]

                    best_response = best_response+best_new_generation
                    eos_flag = False
                    if Output_stream:
                        logger.info(best_new_generation)
                    continue

            new_generation_list = [new_generations[model_name].replace("�"," ") for model_name in model_list]

            stop_spaces_list = [stop_space[model_name] for model_name in model_list]
            
     
            results_score = self.Parallel_verify(new_generation_list,best_response,verify_model_container)
                   
            Eliminate_self_judgment = False
            if Eliminate_self_judgment:
                for m in results_score:
                    results_score[m][model_list.index(m)]=0
            
            ##Hedge update

            if self.Use_Hedge:

                self.eta = self.learn_rate*np.sqrt(1/self.K)/turns_num
                score_list = [results_score[model_name] for model_name in model_list]
                score_list = np.array(score_list)
                
                #use win rate as reward
                wins, rewards = self.compute_winrate_reward(score_list)

                self.update(rewards)
                weighted_scores = score_list * self.w[:, np.newaxis]  # shape: (self_K, self_K)

                summed = weighted_scores.sum(axis=0) 

            else:
                summed = [sum(values) for values in zip(*results_score.values())]

            indexed_sorted = sorted(enumerate(summed), key=lambda x: x[1], reverse=True)
            
            hit_model_id =  indexed_sorted[0][0]

            hit_list.append(self.model_id_dic[model_list[hit_model_id]])
 
            model_hit_times[model_list[hit_model_id]]+=1

            best_new_generation = new_generation_list[hit_model_id]

            if best_new_generation==" .":
                best_response=best_response
            else:
                best_response = best_response+best_new_generation



            if stop_spaces_list[hit_model_id]:
                eos_flag =False

            if Output_stream:
                logger.info(best_new_generation)

        self.Clear_Cache()
        self.w = np.ones(self.K) / self.K
        logger.info(str(model_hit_times))
        return best_response

    

    def Parallel_Inference(self,input_text,inference_models):
        results = {}
        stop_spaces ={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inference_models)) as executor:
            # Submit all tasks and record future objects

            futures = []

            for model in inference_models:
                futures.append(executor.submit(model.Speculative_generate, input_text,self.each_step_max_generate_token))
            
            # Wait for all tasks to complete and get the results
            for future in concurrent.futures.as_completed(futures):
                
                result,model_name,if_finish = future.result()
                results[model_name] = result
                stop_spaces[model_name] = if_finish
           

        return results,stop_spaces
    


    def Parallel_verify(self,input_text_list,best_response,verify_models):
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(verify_models)) as executor:
            # Submit all tasks and record future objects
            futures = []
            for model in verify_models:
                # futures.append(executor.submit(model.model_verify, input_text_list))
                futures.append(executor.submit(model.model_verify_in_line, input_text_list,best_response))

            # Wait for all tasks to complete and get the results
            for future in concurrent.futures.as_completed(futures):
                
                scores,model_name = future.result()
                results[model_name] = scores

        return results
    

    def Clear_Cache(self):
        for model in set(self.inference_model_lists+self.verify_model_lists):
            model.Clear_Cache()
        torch.cuda.empty_cache()

    def Find_eos_token(self,new_generation_list=None,indices=None,Stop_token=None):
        for indice in indices:
            for stop in Stop_token:
                if stop in new_generation_list[indice.item()]:
                    
                    return new_generation_list[indice.item()]
        return None

    
    def cn_str_count(self,str):
        num = 0 
        for s in str:
     
            if '\u4e00' <= s <= '\u9fff':
                num+=1
        return num
    
    def find_if_has_eos_value(self,d):

        return next((k for k, v in d.items() if v is True), None)
    

    def update(self, rewards):
        self.w = self.w * np.exp(self.eta * np.array(rewards))
 
        self.w = self.w / np.sum(self.w)
    

    def compute_winrate_reward(self,scores):
        K = scores.shape[0]
        wins = np.zeros(K, dtype=int)
        for i in range(K):
            for j in range(K):
                if j == i:
                    continue
                for k in range(K):
                    if k == i:
                        continue
                    if scores[j, i] >= scores[j, k]:
                        wins[i] += 1
            wins[i] = max(wins[i], 1)  

        total_comparisons = (K - 1) * (K - 1)
        rewards = wins / total_comparisons
        

        return wins, rewards
