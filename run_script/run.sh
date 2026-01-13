#!/bin/bash




# MODEL_Name_List = [
#     "glm4_9b_instruct",
#     "gemma_2_9b_instruct",
#     "llama3_8b_instruct",
#     "llama3_70b_instruct",
#     "llama3.1_8b_instruct",
#     "llama3.3_70b_instruct",
#     "mistral_7b_instruct_v3",
#     "mistral-small-24b-instruct-2501",
#     "qwen2_7b_instruct",
#     "qwen2_72b_instruct_int8",
#     "qwen2.5_7b_instruct",
#     "qwen2.5_32b_instruct_int8",
# ]



each_step_max_generate_token=10

inference_model_names="llama3_8b_instruct,mistral_7b_instruct_v3,qwen2_7b_instruct"
verify_model_names="llama3_8b_instruct,mistral_7b_instruct_v3,qwen2_7b_instruct"

result_out_folder=""

inference_model_names_for_file=$(echo "$inference_model_names" | sed 's/instruct//g')
verify_model_names_for_file=$(echo "$verify_model_names" | sed 's/instruct//g')

test_file=""


name=""

result_out_file_name="${name}.json"
echo "$result_out_file_name"


log_out_folder=""
log_file_name="${name}.log"

if [ ! -d "$log_out_folder" ]; then
    mkdir -p "$log_out_folder"
    echo " $log_out_folder "
else
    echo "create $log_out_folder "
fi



CUDA_VISIBLE_DEVICES=3,4,5,7  python main.py \
                        --inference_models "$inference_model_names" \
                        --verify_models "$verify_model_names" \
                        --window_size $each_step_max_generate_token \
                        --max_words 3000 \
                        --test_data "$test_file" \
                        --out_folder  "$result_out_folder" \
                        --out_file_name "$result_out_file_name"

