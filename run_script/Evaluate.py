import json
import argparse
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, "..")) 
if project_root not in sys.path:
    sys.path.append(project_root)

from metric import calculate_score,calculate_score_cn
from utils.logger_config import setup_logger

logger = setup_logger(__name__)



def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for the model configuration.")

    parser.add_argument(
        '--test_file', type=str, default="",
        help='Test data path.'
    )
    parser.add_argument(
        '--lang', type=str, default="en",
        help='The language of test data.'
    )

    return parser.parse_args()


def read_file(file):

    reference = []
    model_out = []

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)

            if  "spec_pred" in line:
                out = line["spec_pred"]
            elif "gf_pred" in line:
                out = line["gf_pred"]
            elif "unite_pred" in line:
                out = line["unite_pred"]
            else:
                raise ValueError("No valid prediction found in the line.")
            if out=="":
                continue
                out=" "
            model_out.append(out)
            reference.append(line["output"])
    
    return model_out,reference



def get_file_paths(test_file):
    if os.path.isdir(test_file):
        return [os.path.join(test_file, f) for f in os.listdir(test_file) if os.path.isfile(os.path.join(test_file, f))]
    elif os.path.isfile(test_file):
        return [test_file]
    else:
        raise FileNotFoundError(f"{test_file} does not exist.")


if __name__ == '__main__':

    args = get_args()
    test_file = args.test_file


    file_list = get_file_paths(test_file)

    for file in file_list:

        model_out,reference = read_file(file)

        model_name = os.path.basename(file).strip(".json")
        logger.info("*************************************************************************")
        logger.info(f"The result of {model_name}:")
        if args.lang == "en":
            calculate_score(model_out,reference)

        elif args.lang == "cn":
            calculate_score_cn(model_out,reference,use_jieba=True)



