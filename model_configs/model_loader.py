import torch
from model_configs.config_map import MODEL_CONFIGS
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, "..")) 
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import Model_Factory, ModelConfig



DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto":"auto"
}

def load_models(model_names):
    """
    Load multiple models with dynamically assigned GPU devices.

    Args:
        model_names (list): A list of model names as defined in MODEL_CONFIGS.

    Returns:
        list: A list of initialized model instances.

    Raises:
        RuntimeError: If no GPU is available or if requested models exceed available GPUs.
        ValueError: If any model name is not found in MODEL_CONFIGS.
    """
    inference_model_list = []

    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count == 0:
        raise RuntimeError("No available GPUs detected. Please ensure CUDA is properly configured.")

    if len(model_names) > available_gpu_count:
        raise RuntimeError(
            f"Requested {len(model_names)} models, but only {available_gpu_count} GPUs are available."
        )

    for idx, model_name in enumerate(model_names):
        config = MODEL_CONFIGS.get(model_name)
        if config is None:
            raise ValueError(f"Model config not found for: {model_name}")

        dtype_str = config.get("torch_dtype")
        torch_dtype = DTYPE_MAP.get(dtype_str.lower())
        if torch_dtype is None:
            raise ValueError(f"Invalid torch_dtype '{dtype_str}' in config for model: {model_name}")

        device = torch.device(f"cuda:{idx}")
        config_with_device = {
            **config,
            "device_id": device,
            "torch_dtype": torch_dtype
        }

        model = Model_Factory(model_config=ModelConfig(**config_with_device))
        inference_model_list.append(model)

    return inference_model_list