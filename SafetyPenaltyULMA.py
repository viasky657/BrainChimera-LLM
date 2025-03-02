"""
SafetyPenaltyULMA.py - Implementation of the ULMA-inspired safety penalty for COCONUT models

This module implements a piecewise negative reward function that penalizes the model for selecting
harmful actions, based on the Unified Language Model Alignment (ULMA) approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np

class SafetyIndicator(nn.Module):
    """
    Class to compute the z_{safety_indicator} value in the safety penalty
    This determines if a response is safe (z close to 1) or unsafe (z close to 0)
    """
    def __init__(self, 
                hidden_dim: int = 768, 
                classifier_type: str = "heuristic", 
                threshold: float = 0.5):
        """
        Initialize the safety indicator
        
        Args:
            hidden_dim: Hidden dimension for the classifier (if using neural classifier)
            classifier_type: Type of classifier to use ('heuristic', 'neural', or 'constant')
            threshold: Threshold for binary classification
        """
        super().__init__()
        self.classifier_type = classifier_type
        self.threshold = threshold
        
        # Initialize neural classifier if needed
        if classifier_type == "neural":
            self.safety_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        elif classifier_type == "heuristic":
            # Load harmful keywords/patterns for heuristic matching
            self.harmful_patterns = self._load_harmful_patterns()
        elif classifier_type == "constant":
            # Constant mode just uses a fixed z value (mostly for testing)
            self.constant_z = 0.5
        
    def _load_harmful_patterns(self) -> List[str]:
        """Load harmful patterns/keywords used for heuristic classification"""
        # This is a simplified list - in a real system you would have a more comprehensive database
        return [
            "kill", "harm", "illegal", "weapon", "suicide", "bomb", 
            "steal", "hack", "virus", "malware", "exploit", "attack",
            "hurt", "dangerous", "violent", "toxic", "hate", "racist",
            "sexually explicit", "pornographic", "abusive"
        ]
    
    def forward(self, 
               prompt: str, 
               response: str, 
               embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the safety indicator value
        
        Args:
            prompt: The input prompt
            response: The model's response
            embeddings: Optional tensor of embeddings (for neural classifier)
            
        Returns:
            z_safety_indicator: Value between 0 and 1 (1 = safe, 0 = unsafe)
        """
        if self.classifier_type == "neural" and embeddings is not None:
            # Use neural classifier when embeddings are provided
            safety_score = self.safety_classifier(embeddings).squeeze(-1)
            return safety_score
        
        elif self.classifier_type == "heuristic":
            # Simple heuristic based on keyword/pattern matching
            response_lower = response.lower()
            
            # Count matches with harmful patterns
            match_count = sum(pattern in response_lower for pattern in self.harmful_patterns)
            
            # Convert to safety score (inversely related to match count)
            # More harmful matches = lower safety score
            safety_score = torch.tensor(max(0, 1.0 - 0.2 * match_count), dtype=torch.float32)
            return safety_score
            
        elif self.classifier_type == "constant":
            # Return constant value (useful for testing)
            return torch.tensor(self.constant_z, dtype=torch.float32)
        
        # Default fallback to moderate safety
        return torch.tensor(0.5, dtype=torch.float32)

class SafetyPenaltyULMA(nn.Module):
    """
    Implementation of the ULMA-inspired safety penalty
    R_{safety_penalty_ULMA}(s, a) = -w_{safety_ulma} * (1 - z_{safety_indicator}) * 
                                    log(1 - σ(β_{ulma} * log(πθ(y|x) / πref(y|x)) + β_{ulma} * logZ(x)))
    """
    def __init__(self, 
                w_safety_ulma: float = 1.0, 
                beta_ulma: float = 5.0,
                safety_indicator: Optional[nn.Module] = None,
                hidden_dim: int = 768,
                reference_model: Optional[Any] = None,
                eps: float = 1e-6):
        """
        Initialize the safety penalty
        
        Args:
            w_safety_ulma: Weight for the safety penalty
            beta_ulma: Regularization strength for ULMA component
            safety_indicator: Optional module to compute safety indicator (z)
            hidden_dim: Hidden dimension for internal modules
            reference_model: Optional reference model for computing πref(y|x)
            eps: Small value to prevent log(0)
        """
        super().__init__()
        self.w_safety_ulma = w_safety_ulma
        self.beta_ulma = beta_ulma
        self.eps = eps
        
        # Initialize safety indicator if not provided
        if safety_indicator is None:
            self.safety_indicator = SafetyIndicator(hidden_dim=hidden_dim)
        else:
            self.safety_indicator = safety_indicator
            
        # Store reference model if provided
        self.reference_model = reference_model
        
    def compute_log_probs(self, 
                         model_output_logits: torch.Tensor,
                         target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of the model's output
        
        Args:
            model_output_logits: Logits from model (batch_size, seq_len, vocab_size)
            target_ids: Target token IDs (batch_size, seq_len)
            
        Returns:
            log_probs: Log probabilities (batch_size, seq_len)
        """
        # Apply log softmax to get log probabilities
        log_probs = F.log_softmax(model_output_logits, dim=-1)
        
        # Gather log probs for the target tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return gathered_log_probs
    
    def forward(self, 
               model_output_logits: torch.Tensor,
               target_ids: torch.Tensor,
               prompts: List[str],
               responses: List[str],
               embeddings: Optional[torch.Tensor] = None,
               reference_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the ULMA-inspired safety penalty
        
        Args:
            model_output_logits: Logits from policy model πθ (batch_size, seq_len, vocab_size)
            target_ids: Target token IDs (batch_size, seq_len)
            prompts: List of prompt strings
            responses: List of response strings
            embeddings: Optional embeddings for safety classification
            reference_logits: Optional logits from reference model πref
            
        Returns:
            safety_penalty: The computed safety penalty
        """
        batch_size = model_output_logits.shape[0]
        device = model_output_logits.device
        
        # 1. Calculate log(πθ(y|x))
        log_pi_theta = self.compute_log_probs(model_output_logits, target_ids)
        
        # 2. Calculate log(πref(y|x)) if reference_logits provided, else use approximation
        if reference_logits is not None and self.reference_model is not None:
            log_pi_ref = self.compute_log_probs(reference_logits, target_ids)
        else:
            # Approximate reference model log probs (e.g., use detached policy model)
            log_pi_ref = self.compute_log_probs(model_output_logits.detach(), target_ids)
        
        # 3. Compute log(πθ(y|x) / πref(y|x))
        log_ratio = log_pi_theta - log_pi_ref
        
        # 4. Apply β_{ulma} scaling and logZ(x) term (often approximated to 0)
        # We skip logZ(x) as it's commonly approximated to 0 in practice
        scaled_log_ratio = self.beta_ulma * log_ratio
        
        # 5. Apply sigmoid and compute log(1 - σ(...))
        sigmoid_term = torch.sigmoid(scaled_log_ratio)
        log_term = torch.log(1 - sigmoid_term + self.eps)
        
        # 6. Compute safety indicators (z) for each example in batch
        z_safety_indicators = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            # Get z_safety_indicator for this example
            if embeddings is not None:
                z = self.safety_indicator(prompts[i], responses[i], embeddings[i])
            else:
                z = self.safety_indicator(prompts[i], responses[i])
            z_safety_indicators[i] = z
        
        # 7. Compute final safety penalty
        # R_{safety_penalty_ULMA}(s, a) = -w_{safety_ulma} * (1 - z_{safety_indicator}) * log(1 - σ(...))
        safety_switch = 1 - z_safety_indicators
        safety_penalty = -self.w_safety_ulma * safety_switch.unsqueeze(-1) * log_term
        
        # Average over sequence length and batch
        safety_penalty = safety_penalty.mean()
        
        return safety_penalty

class SafetyDatasetHandler:
    """
    Handler for safety datasets used in training
    
    This class handles loading and processing safety datasets like Anthropic_HH_Golden
    and Aegis, which contain pairs of chosen (safe) and rejected (unsafe) responses.
    These datasets are specifically designed for training models to avoid harmful outputs.
    """
    def __init__(self, dataset_name: str = "Unified-Language-Model-Alignment/Anthropic_HH_Golden"):
        """
        Initialize the safety dataset handler
        
        Args:
            dataset_name: Name of the dataset to load. Should be either 
                         "Unified-Language-Model-Alignment/Anthropic_HH_Golden" or
                         "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
        """
        self.dataset_name = dataset_name
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Any:
        """
        Load the safety dataset
        
        Returns:
            The loaded dataset
        """
        try:
            return load_dataset(self.dataset_name)
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")
            return None
    
    def get_batch(self, batch_size: int, split: str = "train") -> Dict[str, List[str]]:
        """
        Get a batch of examples from the dataset
        
        Args:
            batch_size: Number of examples to include in batch
            split: Dataset split to use ('train', 'validation', 'test')
            
        Returns:
            Dictionary with 'input', 'chosen', 'rejected' lists
        """
        if self.dataset is None:
            return {"prompts": [], "chosen": [], "rejected": []}
        
        ds_split = self.dataset[split]
        indices = torch.randperm(len(ds_split))[:batch_size]
        
        batch = {
            "prompts": [],
            "chosen": [],
            "rejected": []
        }
        
        for idx in indices:
            example = ds_split[int(idx)]
            
            # Handle different dataset formats
            if "chosen" in example and "rejected" in example:
                # Anthropic HH format
                batch["prompts"].append(example.get("input", ""))
                batch["chosen"].append(example.get("chosen", ""))
                batch["rejected"].append(example.get("rejected", ""))
            else:
                # Generic format - adapt as needed for specific datasets
                batch["prompts"].append(example.get("prompt", ""))
                # Assume first completion is chosen, second is rejected if available
                completions = example.get("completions", ["", ""])
                batch["chosen"].append(completions[0] if len(completions) > 0 else "")
                batch["rejected"].append(completions[1] if len(completions) > 1 else "")
        
        return batch

def preprocess_for_safety_training(
    model,
    tokenizer,
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str],
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Preprocess data for safety training
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        chosen_responses: List of chosen (safe) response strings
        rejected_responses: List of rejected (unsafe) response strings
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tensors for training
    """
    batch_size = len(prompts)
    device = next(model.parameters()).device
    
    # Tokenize prompts, chosen and rejected responses
    prompt_tokens = tokenizer(
        prompts, 
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    chosen_tokens = tokenizer(
        chosen_responses,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    rejected_tokens = tokenizer(
        rejected_responses,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    # Get model outputs for chosen and rejected responses
    with torch.no_grad():
        chosen_outputs = model(**prompt_tokens, labels=chosen_tokens.input_ids)
        rejected_outputs = model(**prompt_tokens, labels=rejected_tokens.input_ids)
    
    return {
        "prompts": prompts,
        "chosen_responses": chosen_responses,
        "rejected_responses": rejected_responses,
        "prompt_input_ids": prompt_tokens.input_ids,
        "prompt_attention_mask": prompt_tokens.attention_mask,
        "chosen_input_ids": chosen_tokens.input_ids,
        "chosen_attention_mask": chosen_tokens.attention_mask,
        "rejected_input_ids": rejected_tokens.input_ids,
        "rejected_attention_mask": rejected_tokens.attention_mask,
        "chosen_logits": chosen_outputs.logits,
        "rejected_logits": rejected_outputs.logits,
    }

# Import play_sound function for training completion notification
import os
import sys
import subprocess
import platform

def play_sound(sound_file):
    """
    Play a sound file using the appropriate system command.
    
    Args:
        sound_file: Path to the sound file to play
    """
    try:
        if platform.system() == "Linux":
            subprocess.run(["aplay", sound_file])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", sound_file])
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
        print(f"Sound played: {sound_file}")
    except Exception as e:
        print(f"Failed to play sound: {e}")

def train_safety_penalty(
    model,
    tokenizer,
    safety_penalty: SafetyPenaltyULMA,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    dataset_name: str = "Unified-Language-Model-Alignment/Anthropic_HH_Golden",
    max_length: int = 512,
    sound_file: str = "Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav"
) -> Dict[str, List[float]]:
    """
    Train the model using the safety penalty
    
    Args:
        model: The language model to train
        tokenizer: Tokenizer for the model
        safety_penalty: The SafetyPenaltyULMA module
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        dataset_name: Name of the dataset to use
        max_length: Maximum sequence length
        sound_file: Path to sound file to play when training completes
        
    Returns:
        Dictionary with training metrics
    """
    # Initialize dataset handler
    dataset_handler = SafetyDatasetHandler(dataset_name)
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training metrics
    metrics = {
        "total_loss": [],
        "safety_penalty_chosen": [],
        "safety_penalty_rejected": []
    }
    
    print(f"Starting safety penalty training with {dataset_name}...")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    # Training loop
    model.train()
    safety_penalty.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Get batch of examples
        batch = dataset_handler.get_batch(batch_size, split="train")
        
        # Preprocess data
        processed_data = preprocess_for_safety_training(
            model, 
            tokenizer,
            batch["prompts"],
            batch["chosen"],
            batch["rejected"],
            max_length
        )
        
        # Forward pass for chosen responses
        outputs_chosen = model(
            input_ids=processed_data["prompt_input_ids"],
            attention_mask=processed_data["prompt_attention_mask"],
            labels=processed_data["chosen_input_ids"]
        )
        
        # Forward pass for rejected responses
        outputs_rejected = model(
            input_ids=processed_data["prompt_input_ids"],
            attention_mask=processed_data["prompt_attention_mask"],
            labels=processed_data["rejected_input_ids"]
        )
        
        # Compute safety penalties
        safety_penalty_chosen = safety_penalty(
            outputs_chosen.logits,
            processed_data["chosen_input_ids"],
            batch["prompts"],
            batch["chosen"]
        )
        
        safety_penalty_rejected = safety_penalty(
            outputs_rejected.logits,
            processed_data["rejected_input_ids"],
            batch["prompts"],
            batch["rejected"]
        )
        
        # Total loss: language modeling loss + safety penalty
        # Increase penalty for rejected responses, reduce for chosen responses
        total_loss = (
            outputs_chosen.loss + outputs_rejected.loss +
            0.1 * safety_penalty_chosen - 0.5 * safety_penalty_rejected
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics["total_loss"].append(total_loss.item())
        metrics["safety_penalty_chosen"].append(safety_penalty_chosen.item())
        metrics["safety_penalty_rejected"].append(safety_penalty_rejected.item())
        
        print(f"  Loss: {total_loss.item():.4f}")
        print(f"  Safety Penalty (Chosen): {safety_penalty_chosen.item():.4f}")
        print(f"  Safety Penalty (Rejected): {safety_penalty_rejected.item():.4f}")
    
    # Play sound to indicate training completion
    print("Safety penalty training completed!")
    if os.path.exists(sound_file):
        play_sound(sound_file)
    else:
        print(f"Sound file not found: {sound_file}")
    
    return metrics
