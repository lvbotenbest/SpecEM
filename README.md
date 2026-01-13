# SpecEM: Speculative Ensemble Model


SpecEM is an advanced framework for **ensemble generation** with Large Language Models (LLMs). 

## 🚀 Key Features

### 🎯 Speculative Decoding
### 🤝 Ensemble Generation
### ⚙️ Advanced Configuration


## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for optimal performance

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SpecEM.git
   cd SpecEM
   ```

2. **Install dependencies**
   ```bash
   pip install torch transformers
   pip install auto-gptq bert-score rouge jieba sacrebleu
   pip install tqdm numpy
   ```

3. **Configure model paths**
   Edit `model_configs/config_map.py` to specify your model paths:
   ```python
   "llama3_8b_instruct": {
       "model_path": "/path/to/your/llama3-8b-instruct",
       "torch_dtype": "bfloat16"
   }
   ```

## 🎯 Quick Start

### Basic Usage

Run inference with a text prompt:

```bash
# Generate text with multiple models
python demo.py \
  --inference_models "llama3_8b_instruct,mistral_7b_instruct_v3" \
  --verify_models "llama3_8b_instruct,mistral_7b_instruct_v3" \
  --out_stream True \
  --user_input "What is the future of artificial intelligence?"


cd run_script

# For file processing, use main.py instead
python main.py \
  --inference_models "llama3_8b_instruct,mistral_7b_instruct_v3" \
  --verify_models "llama3_8b_instruct,mistral_7b_instruct_v3" \
  --test_data "../input" \
  --out_folder "../output" \
  --out_file_name "results.json"

# Evaluate the generated results
python Evaluate.py \
  --test_file "../output/results.json" \
  --lang "en"
```

### Advanced Configuration

```bash
python main.py \
  --inference_models "llama3_8b_instruct,qwen2_7b_instruct,mistral_7b_instruct_v3" \
  --verify_models "llama3_8b_instruct,qwen2_7b_instruct,mistral_7b_instruct_v3" \
  --window_size 15 \
  --max_words 2000 \
  --fast_decoder True \
  --out_stream False \
  --test_data "../test_data.txt" \
  --out_folder "../results" \
  --out_file_name "advanced_results.json"
```

## 📚 Configuration Reference

### Model Parameters

#### demo.py Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--inference_models` | str | "llama3_8b_instruct,mistral_7b_instruct_v3" | Comma-separated inference models |
| `--verify_models` | str | "glm4_9b_instruct" | Comma-separated verification models |
| `--user_input` | str | "" | Text prompt for generation |
| `--window_size` | int | 10 | Segment max length for SpecFuse |
| `--max_words` | int | 1500 | Maximum words per query |
| `--fast_decoder` | bool | False | Enable fast decoding mode |
| `--out_stream` | bool | False | Enable output streaming |

#### main.py Parameters (File Processing)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--test_data` | str | "" | Path to input data file |
| `--out_folder` | str | "" | Output directory |
| `--out_file_name` | str | "" | Output filename |

### Supported Models

| Model | Size | Recommended Use |
|-------|------|-----------------|
| LLaMA-3 8B Instruct | 8B | Fast inference |
| LLaMA-3 70B Instruct | 70B | High-quality verification |
| Mistral 7B v3 | 7B | Efficient drafting |
| Qwen2 7B/72B Instruct | 7B/72B | Multilingual support |
| GLM-4 9B Instruct | 9B | Balanced performance |
| Gemma-2 9B Instruct | 9B | Research applications |

## 🏗️ Architecture

### Core Components

```
SpecEM/
├── 📁 metric/                 # Evaluation metrics
│   ├── 📄 score.py           # BLEU, ROUGE, BERTScore
│   └── 📁 huggingface_bleu/  # BLEU implementation
├── 📁 model_configs/          # Model configurations
│   ├── 📄 config_map.py      # Model registry
│   └── 📄 model_loader.py    # Dynamic loading
├── 📁 run_script/             # Execution scripts
│   ├── 📄 demo.py           # Main inference pipeline
│   ├── 📄 Evaluate.py       # Evaluation script
│   └── 📄 main.py           # Advanced execution
└── 📁 utils/                  # Core utilities
    ├── 📄 normal_generate.py # EnsembleService class
    ├── 📄 normal_utils.py    # Model factory
    └── 📄 logger_config.py   # Logging system
```

### Workflow

1. **Model Loading**: Dynamically load inference and verification models
2. **Draft Generation**: Multiple models generate candidate segments
3. **Verification**: Verification models validate and select best segments
4. **Ensemble**: Combine verified segments using weighted voting
5. **Output**: Generate final text with quality metrics



### Optimization Tips

- **Use smaller models** for inference when speed is critical
- **Larger verification models** improve quality significantly
- **Adjust window size** based on text length (5-15 optimal)
- **Enable fast decoder** for real-time applications

## 🔧 Advanced Usage

### Custom Model Integration

Add your own models to `model_configs/config_map.py`:

```python
MODEL_CONFIGS["your_model"] = {
    "model_type": "your_model_type",
    "model_path": "/path/to/your/model",
    "torch_dtype": "bfloat16"  # or "float16", "auto"
}
```


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{lvspecem,
  title={SpecEM: Training-Free LLM Ensembling via Iterative Drafting, Verification, and Online Feedback},
  author={Lv, Bo and Liu, Nayu and Tang, Chen and Liu, Xin and Yu, Yue and Luo, Ping},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
