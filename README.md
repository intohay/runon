# RUNON (Relative Uniqueness using Normalized Offset to Non-personalized model)

This repository contains the implementation of RUNON (Relative Uniqueness of Next-token Occurrence Novelty), a metric for evaluating persona-likeness in language model outputs, along with human evaluation datasets and correlation analysis scripts.

## Overview

RUNON is a reference-free evaluation metric that measures how well a language model captures specific personas by comparing the token probability distributions between a fine-tuned persona model and a base model.

## Repository Structure

```
runon/
├── scripts/
│   ├── train.py              # Training script for persona models
│   ├── evaluate.py           # Evaluation script supporting multiple metrics including RUNON
│   └── calculate_spearman_correlation.py  # Script to calculate correlation with human evaluations
├── src/
│   ├── model/               # Model loading and configuration
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation metrics
│   │   └── lunon_metric.py  # RUNON implementation
│   └── data/               # Data loading utilities
├── data/
│   └── evaluation/         # Evaluation datasets
│       ├── A/              # Persona A evaluation data
│       ├── B/              # Persona B evaluation data
│       ├── C/              # Persona C evaluation data
│       ├── D/              # Persona D evaluation data
│       └── E/              # Persona E evaluation data
├── README.md
└── LICENSE
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- Hydra
- Pandas
- NumPy
- SciPy

## Installation

```bash
pip install torch transformers unsloth hydra-core pandas numpy scipy
```

## Usage

### Training Persona Models

```bash
python scripts/train.py \
    --config-path ../configs \
    --config-name train \
    model.model_name=tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
    data.train_file=path/to/train.jsonl \
    data.eval_file=path/to/eval.jsonl
```

### Evaluating with RUNON

```bash
python scripts/evaluate.py \
    --config-path ../configs \
    --config-name evaluate \
    evaluation.metrics=[lunon] \
    evaluation.lunon.ft_model_name=path/to/finetuned/model \
    evaluation.test_file=data/evaluation/A/for_evaluation2_shuffled.jsonl
```

### Calculating Correlation with Human Evaluations

```bash
python scripts/calculate_spearman_correlation.py \
    data/evaluation/A/human_evaluation_scores.csv \
    path/to/system_evaluation_results.csv
```

## Data Format

### Training Data
Training data should be in JSONL format with each line containing:
```json
{"text": "Persona dialogue or monologue text"}
```

**Note**: Training data links will be made publicly available upon publication. Please check back for updates.

### Evaluation Data
Evaluation data should be in JSONL format with:
```json
{"prefix": "Context or prompt", "continuation": "Expected persona response"}
```

### Human Evaluation Data
Human evaluation scores are provided in CSV format with columns:
- `respondent_id`: Anonymized evaluator ID
- `question_num`: Question/sample number
- `score`: Likert scale score (1-5)
- `weighted_score`: Score weighted by evaluator reliability

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{runon2025,
  title={RUNON: A Reference-Free Metric for Evaluating Persona-Likeness in Language Models},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Privacy and Ethics

All human evaluation data has been anonymized to protect participant privacy. Persona names have been replaced with letters (A-E) for confidentiality.
