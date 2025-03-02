# CROW Backdoor Elimination with RedPajama Dataset

This project implements the CROW (Consistency Regularization for backdOor elimination in Weights) method for eliminating backdoors from large language models. This implementation includes integration with the RedPajama dataset from Hugging Face for clean data training.

## Overview

The CROW method reduces adversarial (poison prompt) attacks by approximately 65% through adversarial fine-tuning that enforces smooth transitions across transformer layers. This implementation provides:

- A robust backdoor elimination technique for language models
- Integration with the RedPajama dataset, a high-quality dataset for model training
- Character-limited dataset loading to control training size
- Support for both direct dataset usage and automated dataset handling

## Key Components

### Files

- `Crow.py`: Core implementation of the CROW backdoor elimination technique
- `redpajama_example.py`: Example script demonstrating how to use CROW with RedPajama for model purification
- `test_redpajama_loading.py`: Simple test script to verify RedPajama dataset loading

### Features

- **Adversarial Perturbation Generation**: Simulates backdoor triggers by generating adversarial examples on input embeddings
- **Adversarial Consistency Training**: Ensures that even with perturbed inputs, the model's hidden state transitions remain consistent
- **RedPajama Integration**: Uses the high-quality RedPajama dataset with character-limited loading (1,000 character limit by default)
- **Flexible Dataset Handling**: Support for both pre-loading datasets and automatic dataset creation

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from Crow import apply_crow_training

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Apply CROW training with RedPajama dataset
purified_model, metrics = apply_crow_training(
    model=model,
    use_redpajama=True,
    tokenizer=tokenizer
)

# Your model is now purified of potential backdoors
```

### Advanced Usage

For more control over the training process:

```python
from Crow import CROWBackdoorElimination, get_redpajama_dataset

# Load RedPajama dataset with custom character limit
redpajama_dataset = get_redpajama_dataset(max_chars=1000)

# Prepare for your model
tokenized_dataset = redpajama_dataset.prepare_for_model(tokenizer)

# Initialize CROW trainer with custom parameters
crow_trainer = CROWBackdoorElimination(
    model=model,
    epsilon=0.1,  # Perturbation magnitude
    alpha=5.5,    # Consistency weight
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=4,
    device='cuda'
)

# Apply CROW training
metrics = crow_trainer.train(tokenized_dataset)
```

## Testing

To verify the RedPajama dataset loading without running the full training process:

```bash
python test_redpajama_loading.py
```

This script will load a small sample of the dataset and display basic information to confirm that the integration is working correctly.

## Technical Details

The CROW method works through:

1. **Layer-wise Consistency Regularization**: Enforces smooth transitions between transformer layers
2. **Adversarial Perturbation**: Uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples
3. **Cosine Similarity Measurement**: Quantifies the consistency between consecutive hidden states

## Hyperparameters

Key hyperparameters for the CROW method:

- **Epsilon (ε)**: Controls the magnitude of adversarial perturbations (default: 0.1)
- **Alpha (α)**: Weighting factor for consistency regularization (default: 5.5, higher for targeted refusal tasks: 11.0)
- **Learning Rate**: Rate for parameter updates (default: 2e-5)
- **Batch Size**: Training batch size (default: 4)
- **Number of Epochs**: Training iterations (default: 3)

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)

## RedPajama Dataset

This implementation uses the RedPajama dataset from Hugging Face (`togethercomputer/RedPajama-Data-V2`), specifically:
- Partition: "head_middle"
- Snapshot: "2022-49"
- Language: "en"

The dataset is limited to 1,000 characters by default to provide sufficient but manageable training data.
