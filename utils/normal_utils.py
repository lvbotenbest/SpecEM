from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class ModelConfig:
    def __init__(
            self, 
            model_type: str = "",
            model_path: str = "",
            device_id: str = "",
            torch_dtype="auto",
        ):
        self.model_type: str = model_type
        self.model_path: str = model_path
        self.device_id: str = device_id
        self.torch_dtype = torch_dtype
      
        
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0, -1].item()
        return last_token_id in self.stop_ids


class ModelFactory(object):

    def __init__(self,model_config:ModelConfig):

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path,trust_remote_code=True,)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model_type = model_config.model_type
        self.device = model_config.device_id

        if "qwen_2_72b_instruct_int8" in model_config.model_type:
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_config.model_path,
                device=self.device
        ).eval()    
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_path,
                    torch_dtype=model_config.torch_dtype,
                    load_in_4bit=False,
                    trust_remote_code=True,
        ).eval()
            self.model = self.model.to(self.device)


        self.__previous_id = torch.tensor(0,dtype=torch.long)
        self.ascii_letters_digits = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


        if "gemma" in self.model_type:
            self.stop_ids = [
            self.tokenizer.convert_tokens_to_ids("<eos>"),
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
        elif "llama" in self.model_type:
            self.stop_ids = [
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eom_id|>")
            ]
        elif "qwen" in self.model_type:
            self.stop_ids = [
            self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        elif "mistral" in self.model_type:
            self.stop_ids = [
            self.tokenizer.convert_tokens_to_ids("</s>"),
            ]
        elif "glm4" in self.model_type:
            self.stop_ids = [151329, 151336, 151338]
                     
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_ids)])

# inference generate stage 
    def Speculative_generate(self,input_text,each_step_max_generate_token):
        each_step_max_generate_token = each_step_max_generate_token
        if  not self.__previous_id.shape:
            
            if "mistral_small_24b" in self.model_type:

                input_ids = self.tokenizer.encode(f"[SYSTEM_PROMPT]You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI. \
                                                  Please respond to the user's question or instruction below.[/SYSTEM_PROMPT][INST]{input_text}[/INST]",
                                              add_special_tokens=True,
                                              return_tensors="pt").to(self.device)
                
            elif "llama3_70b" in self.model_type:
                 
                input_ids = self.tokenizer.encode(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                                              add_special_tokens=True,
                                              return_tensors="pt").to(self.device)

            else:
                messages = [{"role": "user", "content": input_text},]

                input_ids = self.tokenizer.apply_chat_template(
                                        messages,
                                        add_generation_prompt=True,
                                        return_tensors="pt"
                                        ).to(self.device)
       
        else:
     
            token_ids = self.tokenizer.encode(input_text,
                                              add_special_tokens=False,
                                              return_tensors="pt").to(self.device)

            input_ids = torch.cat((self.__previous_id,token_ids),1)
            
        self.__previous_id = input_ids
        

        attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=self.device)
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            pad_token_id = self.tokenizer.eos_token_id,
            max_new_tokens=each_step_max_generate_token,
            stopping_criteria=self.stopping_criteria,
            )
        outputs = self.model.generate(**generate_kwargs)
        
        response = outputs[0][input_ids.shape[-1]:]

        return_span = self.tokenizer.decode(response,skip_special_tokens=True)

        if outputs[0][-1].item() in self.stop_ids:
            eos_flag = True
        else:   
            eos_flag = False
        

        return_span = self.postprocess_generated_text(return_span)

        # response.shape[-1]-self.tokenizer.encode(text,add_special_tokens=False)
        # postrocess_text_token_length = self.tokenizer.encode(text,add_special_tokens=False)
        #kvcahe= k[:postrocess_text_token_length]
        return return_span,self.model_type,eos_flag
 



    # This function is used to verify the quality of newly generated text by parallel computing.
    @torch.no_grad()
    def model_verify(self,new_generate):
        
        use_softmax = False
        # tokenize all text including previous text and the newly generated text
        
        previous_text = self.tokenizer.decode(self.__previous_id[0])
        new_updates_text_list = [previous_text+text  for text in new_generate]

        inputs = self.tokenizer.batch_encode_plus(new_updates_text_list,padding=True, return_tensors="pt",add_special_tokens=False).to(self.device)

        # get the length of previous text
        
        length_previous_ids = self.__previous_id.shape[-1]



        position_ids = torch.tensor(0)
        id_length = inputs["attention_mask"].shape[-1]
        for i in range(inputs["attention_mask"].shape[0]):
    
            one_nums = torch.sum(inputs["attention_mask"][i])
            position_id  = [0]*(id_length-one_nums)+list(range(one_nums))
            position_id = torch.LongTensor(position_id).to(self.device).unsqueeze(0)
            
            if not position_ids.shape:
                position_ids = position_id
            else:
                position_ids = torch.cat((position_ids,position_id),0)
       

        with torch.no_grad():
        # feed into the model to get the logits for verify the quality of new generated text
            logits = self.model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],).logits


            if use_softmax:
                probs = F.softmax(logits[:, :-1], dim=-1)  # shape: [1, seq_len-1, vocab_size]
            else:
                probs = logits[:, :-1]

            target_ids = inputs["input_ids"][:, 1:]
                
            target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # shape: [batch_size, seq_len-1]

            probability = []
            for i in range(inputs["attention_mask"].shape[0]):

                att_length = (inputs["attention_mask"][i] == 0).sum(dim=-1).item()
                
                p = target_probs[i][att_length+length_previous_ids-1:].mean(dim=-1).item()
                probability.append(p)


            if not use_softmax:
                # probability = torch.softmax(torch.tensor(probability), dim=0)
                probability = self.normalize_sum_to_one(probability)
            else:
                probability = self.normalize_sum_to_one(probability)
        
        return probability,self.model_type 


    # This function is used to verify the quality of newly generated text by parallel computing.
    @torch.no_grad()
    def model_verify_in_line(self,new_generate,previous_best_response):
        
        use_softmax = False
        # tokenize all text including previous text and the newly generated text

        # get the length of previous text
     
        if not self.__previous_id.shape:
            messages = [{"role": "user", "content": previous_best_response},]

            verify_input_id = self.tokenizer.apply_chat_template(
                                        messages,
                                        add_generation_prompt=True,
                                        return_tensors="pt"
                                        ).to(self.device)
            length_previous_ids = verify_input_id.shape[-1]
            
        else:

            length_previous_ids = self.__previous_id.shape[-1]

            verify_input_id = self.__previous_id


        position_list = []

        passage_start_positions = []  
        passage_end_positions   = []

        

        for text in new_generate:
            
            passage_start_positions.append(verify_input_id.shape[-1])
            text_input_id = self.tokenizer.encode(text,add_special_tokens=False,return_tensors="pt").to(self.device)
            verify_input_id = torch.cat((verify_input_id,text_input_id),1)
            
            position_list.append(length_previous_ids+text_input_id.shape[-1])  #append last token position +1
            passage_end_positions.append(verify_input_id.shape[-1]-1)
    

         # prepare for position_ids
        position_ids = list(range(length_previous_ids))
        assert len(position_list) >1 

        for i in position_list:
            position_ids += list(range(length_previous_ids,i))
        
        position_ids = torch.LongTensor(position_ids).to(self.device).unsqueeze(0)

        att = self.passage_mask_attention_matrix(verify_input_id, passage_start_positions, passage_end_positions, self.model.dtype)


        with torch.no_grad():
        # feed into the model to get the logits for verify the quality of new generated text
            logits = self.model(input_ids=verify_input_id,
                   attention_mask=att.to(self.device),
                   position_ids=position_ids
            ).logits


            if use_softmax:
                probs = F.softmax(logits[:, :-1], dim=-1)  # shape: [1, seq_len-1, vocab_size]
            else:
                probs = logits[:, :-1]


            target_ids = verify_input_id[:, 1:]  

      
            for num in passage_start_positions[1:]:
                probs[:,num-1] = probs[:,length_previous_ids-1]

            target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # shape: [batch_size, seq_len-1]

            probability = []

            for pos in zip(passage_start_positions,passage_end_positions):
                start_pos = pos[0]
                end_pos = pos[1]
                p = target_probs[0][start_pos-1:end_pos].mean(dim=-1).item()
                probability.append(p)

            if not use_softmax:
                # probability = torch.softmax(torch.tensor(probability), dim=0)
                probability = self.normalize_sum_to_one(probability)
            else:
                probability = self.normalize_sum_to_one(probability)
         

        return probability,self.model_type 




    # calculate the mean logits score of each candidate generated text
    def get_verify_score(self,start_position,tokens_id,logits,att_mask):

        att_length = (att_mask == 0).sum(dim=-1).item()

        softmax_logits=F.softmax(logits, dim=-1)
        softmax_logits=softmax_logits.squeeze()[att_length+start_position:-1]
        ids=tokens_id.unsqueeze(-1)[att_length+start_position+1:]
        mean_score=softmax_logits.gather(1,ids).transpose(0,1).squeeze().mean(-1)
        return mean_score   

    def Clear_Cache(self):
        self.__previous_id = torch.tensor(0)

    def Get_length(self):
        
        return self.__previous_id.shape[-1]

        
    def postprocess_generated_text(self,text):
        
        if not text:
            return " ."
        
        if len(text) <= 1:
            return text
        

        stripped_text = text.rstrip()  
 
        if not stripped_text:
            return text
        elif stripped_text[-1] not in self.ascii_letters_digits:  
            return stripped_text
        
        last_space_idx = stripped_text.rfind(' ')
  

        if not stripped_text[:last_space_idx].rstrip():
            return stripped_text

        if "mistral_instruct_v3.0_7b" in self.model_type:
            return " "+stripped_text[:last_space_idx].rstrip()
        
        return stripped_text[:last_space_idx].rstrip() 

    


    def normalize_sum_to_one(self,lst):
        total = sum(lst)
        if total == 0:
            return [0 for _ in lst]  
        return [x / total for x in lst]
    
    def generate_mask(self,N, passage_start_positions, passage_end_positions, dtype, fill_value=-65504):
      

        mask = torch.zeros((N, N), dtype=dtype)
        
        special_start_token = passage_start_positions[0]
        
        for start, end in zip(passage_start_positions, passage_end_positions):
   
            mask[start:end+1, special_start_token:start] = fill_value
        return mask

    def passage_mask_attention_matrix(self,input_id, passage_start_positions, passage_end_positions, model_dtype):

        batch_size = input_id.shape[0]
        seq_length = input_id.shape[-1]
        
 
        min_dtype = torch.finfo(model_dtype).min

        causal_mask = torch.full((seq_length, seq_length), fill_value=min_dtype, dtype=model_dtype)
        if seq_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_length, seq_length]
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        

        passage_mask = self.generate_mask(seq_length, passage_start_positions, passage_end_positions, model_dtype, fill_value=min_dtype)
        passage_mask = passage_mask.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_length, seq_length]
        passage_mask = passage_mask.expand(batch_size, 1, seq_length, seq_length)
        
 
        attention_mask = causal_mask + passage_mask
        return attention_mask
