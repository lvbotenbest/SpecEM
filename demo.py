import torch
import json
from tqdm import tqdm
import os
import sys
import argparse


from model_configs import load_models
from utils import EnsembleService
from utils.logger_config import setup_logger

logger = setup_logger(__name__)





def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for the model configuration.")

    # Test folder
    parser.add_argument(
        '--inference_models', type=str, default="llama3_8b_instruct,mistral_7b_instruct_v3,glm4_9b_instruct,qwen2_7b_instruct",
        help='Comma-separated list of models used for inference.'
    )
    parser.add_argument(
        '--verify_models', type=str, default="glm4_9b_instruct",
        help='Comma-separated list of models used for segment verification.'
    )
    parser.add_argument(
        '--fast_decoder', type=bool, default=False,
        help='Enable fast decoding mode to accelerate generation'
    )
    parser.add_argument(
        '--out_stream', type=bool, default=False,
        help='Enable out of stream.'
    )
    # Max length
    parser.add_argument(
        '--window_size', type=int, default=10,
        help='Segment max length used in the SpecFuse.'
    )
    parser.add_argument(
        '--max_words', type=int, default=1500,
        help='Maximum number of words allowed in a single query.'
    )
    parser.add_argument(
        '--learn_rate', type=int, default=1000,
        help='Maximum number of words allowed in a single query.'
    )
    parser.add_argument(
        '--user_input', type=str, default="",
        help='User input prompt.'
    )



    return parser.parse_args()




def read_previous_data(file):

    instruction_list = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            instruction_list[line['instruction']] = line['dataset']
    return instruction_list


def return_instruction_with_dirs(file_path):

    dir_path = os.path.dirname(file_path)
   
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"dir {dir_path} is built")
    
    if  os.path.exists(file_path):
        instruction_list = read_previous_data(file_path)
    else:
        instruction_list = {}
    return instruction_list


#
MODEL_KEYS = [
    "glm4_9b_instruct",
    "gemma_2_9b_instruct",
    "llama3_8b_instruct",
    "llama3_70b_instruct",
    "llama3.1_8b_instruct",
    "llama3.3_70b_instruct",
    "mistral_7b_instruct_v3",
    "mistral-small-24b-instruct-2501",
    "qwen2_7b_instruct",
    "qwen2_72b_instruct_int8",
    "qwen2.5_7b_instruct",
    "qwen2.5_32b_instruct_int8",
    "qwen2.5_32b_instruct",
    "qwen2_72b_instruct"
]




if __name__ == "__main__":


    args = get_args()
    
    inference_model_names = args.inference_models.split(",")
    verify_model_names = args.verify_models.split(",")  

    inference_model_lists = load_models(args.inference_models.split(","))

    model_map = dict(zip(inference_model_names, inference_model_lists))

    verify_model_lists = []
    for name in verify_model_names:
        if name in model_map:
            verify_model_lists.append(model_map[name])
        else:
            verify_model_lists.append(load_models([name])[0])


    logger.info(f"Inference Models: {inference_model_names}")
    logger.info(f"Verify Models: {verify_model_names}")
    logger.info(f"Verify Model Lists: {verify_model_lists}")
    logger.info(f"Window Size: {args.window_size}")

    each_step_max_generate_token = args.window_size 

    max_tokens = args.max_words

    LLMs = EnsembleService(inference_model_lists,verify_model_lists,max_tokens,each_step_max_generate_token,learn_rate=args.learn_rate)
            
    
    # Define the prompt to be processed
    prompt = args.user_input

    response = LLMs.generate(input_text=prompt,short_decoder=False,Output_stream=args.out_stream)


