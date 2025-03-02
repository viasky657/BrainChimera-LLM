import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import datetime
import numpy as np
from datasets import load_dataset

class CROWBackdoorElimination:
    """
    CROW (Consistency Regularization for backdOor elimination in Weights) implementation.
    
    This class implements the CROW method for eliminating backdoors from large language models
    via internal consistency regularization. The technique can reduce adversarial (poison prompt)
    attacks by approximately 65% through adversarial finetuning that enforces smooth transitions
    across transformer layers.
    
    The training process consists of two main components:
    1. Adversarial perturbation generation - simulates backdoor triggers by generating
       adversarial examples on the input embeddings
    2. Adversarial consistency training - ensures that even with perturbed inputs, the model's
       hidden state transitions remain consistent, thereby neutralizing backdoor effects
    """
    def __init__(self, model, epsilon=0.1, alpha=5.5, learning_rate=2e-5, 
                 warmup_ratio=0.1, batch_size=4, num_epochs=3, device='cuda',
                 targeted_refusal_alpha=11.0):
        """
        Initialize the CROW backdoor elimination module.
        
        Args:
            model: The language model to fine-tune
            epsilon: Perturbation magnitude for adversarial examples (default: 0.1)
            alpha: Weighting factor for consistency regularization (default: 5.5)
                   For targeted refusal tasks, a higher value (e.g., 11.0) can be used
            learning_rate: Learning rate for parameter updates (default: 2e-5)
            warmup_ratio: Ratio of steps for learning rate warmup (default: 0.1)
            batch_size: Training batch size (default: 4)
            num_epochs: Number of training epochs (default: 3) 
            device: Device to perform training on (default: 'cuda')
            targeted_refusal_alpha: Higher alpha value for targeted refusal tasks (default: 11.0)
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.targeted_refusal_alpha = targeted_refusal_alpha
        
        # Track training progress and metrics
        self.training_metrics = {
            'consistency_losses': [],
            'llm_losses': [],
            'total_losses': [],
            'clean_accuracy': [],
            'backdoor_resistance': []
        }
    
    def _get_hidden_states(self, input_ids, input_embeddings=None):
        """
        Get all hidden states from each transformer layer.
        
        Args:
            input_ids: Input token IDs
            input_embeddings: Optional alternative input embeddings to use instead of embedding the input_ids
            
        Returns:
            List of hidden states from each transformer layer
        """
        # This implementation will vary based on the specific model architecture
        # For demonstration, we'll access the hidden states through the model's forward pass
        
        # Store original setting for output_hidden_states
        original_output_hidden_states = self.model.config.output_hidden_states
        self.model.config.output_hidden_states = True
        
        # Forward pass with either input_ids or provided embeddings
        with torch.no_grad():
            if input_embeddings is not None:
                # Use the provided input embeddings directly
                outputs = self.model(inputs_embeds=input_embeddings)
            else:
                # Use the input_ids and let the model create the embeddings
                outputs = self.model(input_ids)
        
        # Restore original setting
        self.model.config.output_hidden_states = original_output_hidden_states
        
        # Return all hidden states
        return outputs.hidden_states
    
    def _compute_consistency_loss(self, hidden_states):
        """
        Compute consistency loss across all transformer layers.
        This measures the deviation between consecutive hidden states using cosine similarity.
        
        Args:
            hidden_states: List of hidden states from each transformer layer
            
        Returns:
            Consistency loss (scalar tensor)
        """
        consistency_loss = 0.0
        num_layers = len(hidden_states) - 1  # Exclude input embeddings
        
        for l in range(1, num_layers):
            # Get consecutive hidden states
            current = hidden_states[l]
            next_state = hidden_states[l+1]
            
            # Normalize the hidden states
            current_norm = torch.nn.functional.normalize(current, p=2, dim=-1)
            next_norm = torch.nn.functional.normalize(next_state, p=2, dim=-1)
            
            # Compute cosine similarity (higher value means more similar)
            similarity = torch.sum(current_norm * next_norm, dim=-1).mean()
            
            # Convert to cosine distance (lower value means more similar)
            cosine_distance = 1.0 - similarity
            
            # Add to consistency loss
            consistency_loss += cosine_distance
        
        # Average over layers
        consistency_loss = consistency_loss / (num_layers - 1)
        
        return consistency_loss
    
    def _generate_adversarial_embeddings(self, input_ids):
        """
        Generate adversarial perturbations for the input embeddings.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Adversarially perturbed input embeddings
        """
        # Get the original input embeddings
        orig_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Create a copy that requires gradients
        input_embeds = orig_embeds.clone().detach().requires_grad_(True)
        
        # Get hidden states for original embeddings
        hidden_states = self._get_hidden_states(None, input_embeds)
        
        # Compute consistency loss
        cons_loss = self._compute_consistency_loss(hidden_states)
        
        # Compute gradient of consistency loss with respect to input embeddings
        cons_loss.backward()
        grad = input_embeds.grad.clone()
        
        # Reset gradients
        input_embeds.grad.zero_()
        
        # Generate adversarial perturbation using Fast Gradient Sign Method (FGSM)
        delta = self.epsilon * torch.sign(grad)
        
        # Create adversarial embeddings
        adv_embeds = orig_embeds + delta
        
        return adv_embeds
    
    def train(self, train_dataset, eval_dataset=None, backdoor_test_dataset=None):
        """
        Apply CROW adversarial consistency training to eliminate backdoors.
        
        Args:
            train_dataset: Clean dataset for fine-tuning (can be small, e.g., 100 samples)
            eval_dataset: Optional evaluation dataset for tracking clean performance
            backdoor_test_dataset: Optional dataset containing backdoor triggers for measuring resistance
            
        Returns:
            Dictionary containing training metrics
        """
        print("Starting CROW Backdoor Elimination training...")
        
        # Set up optimizer with cosine decay and warmup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Set up learning rate scheduler
        total_steps = len(train_dataset) // self.batch_size * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        # Initialize data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            epoch_llm_loss = 0.0
            epoch_cons_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in train_loader:
                global_step += 1
                num_batches += 1
                
                # Get input_ids and labels from batch
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else input_ids.clone()
                
                # Generate adversarial embeddings
                adv_embeds = self._generate_adversarial_embeddings(input_ids)
                
                # Forward pass with adversarial embeddings
                outputs = self.model(inputs_embeds=adv_embeds, labels=labels)
                llm_loss = outputs.loss  # Standard language modeling loss
                
                # Get hidden states for adversarial embeddings
                hidden_states_adv = self._get_hidden_states(None, adv_embeds)
                
                # Compute adversarial consistency loss
                adv_cons_loss = self._compute_consistency_loss(hidden_states_adv)
                
                # Combine losses
                # Use higher alpha for targeted refusal tasks if needed
                if hasattr(batch, 'task_type') and batch['task_type'] == 'targeted_refusal':
                    current_alpha = self.targeted_refusal_alpha
                else:
                    current_alpha = self.alpha
                    
                total_loss = llm_loss + current_alpha * adv_cons_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update learning rate
                if global_step < warmup_steps:
                    # Linear warmup
                    lr_scale = min(1.0, float(global_step) / warmup_steps)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * self.learning_rate
                else:
                    scheduler.step()
                
                # Track losses
                epoch_llm_loss += llm_loss.item()
                epoch_cons_loss += adv_cons_loss.item()
                epoch_total_loss += total_loss.item()
                
                # Log progress
                if global_step % 10 == 0:
                    print(f"Step {global_step}: LLM Loss = {llm_loss.item():.4f}, "
                          f"Cons Loss = {adv_cons_loss.item():.4f}, "
                          f"Total Loss = {total_loss.item():.4f}")
            
            # Compute epoch averages
            epoch_llm_loss /= num_batches
            epoch_cons_loss /= num_batches
            epoch_total_loss /= num_batches
            
            # Record metrics
            self.training_metrics['llm_losses'].append(epoch_llm_loss)
            self.training_metrics['consistency_losses'].append(epoch_cons_loss)
            self.training_metrics['total_losses'].append(epoch_total_loss)
            
            print(f"Epoch {epoch+1} completed: Avg LLM Loss = {epoch_llm_loss:.4f}, "
                  f"Avg Cons Loss = {epoch_cons_loss:.4f}, "
                  f"Avg Total Loss = {epoch_total_loss:.4f}")
            
            # Evaluate if datasets are provided
            if eval_dataset is not None:
                clean_acc = self._evaluate_clean_performance(eval_dataset)
                self.training_metrics['clean_accuracy'].append(clean_acc)
                print(f"Clean performance: {clean_acc:.4f}")
            
            if backdoor_test_dataset is not None:
                backdoor_res = self._evaluate_backdoor_resistance(backdoor_test_dataset)
                self.training_metrics['backdoor_resistance'].append(backdoor_res)
                print(f"Backdoor resistance: {backdoor_res:.4f}")
        
        # Save the CROW-purified model
        self._save_purified_model()
        
        print("CROW Backdoor Elimination training completed successfully!")
        return self.training_metrics
    
    def _evaluate_clean_performance(self, eval_dataset):
        """
        Evaluate the model's performance on clean data.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Accuracy score
        """
        self.model.eval()
        
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else None
                
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                if labels is None:
                    # For language modeling: use next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    preds = torch.argmax(shift_logits, dim=-1)
                    correct = (preds == shift_labels).float().sum().item()
                    total = shift_labels.numel()
                else:
                    # For classification tasks
                    preds = torch.argmax(logits, dim=-1)
                    correct = (preds == labels).float().sum().item()
                    total = labels.numel()
                
                total_correct += correct
                total_samples += total
        
        self.model.train()
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def _evaluate_backdoor_resistance(self, backdoor_dataset):
        """
        Evaluate the model's resistance to backdoor triggers.
        
        Args:
            backdoor_dataset: Dataset containing backdoor triggers
            
        Returns:
            Resistance score (higher is better)
        """
        self.model.eval()
        
        backdoor_loader = torch.utils.data.DataLoader(
            backdoor_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # For resistance, we want to measure how often the model does NOT perform the backdoor behavior
        # This is often task-specific, so we'll use a placeholder scoring function
        resistance_score = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in backdoor_loader:
                input_ids = batch["input_ids"].to(self.device)
                backdoor_targets = batch.get("backdoor_targets", None)
                
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # Different backdoor types require different evaluation metrics
                if "backdoor_type" in batch:
                    if batch["backdoor_type"] == "sentiment_steering":
                        # For sentiment steering backdoors: check if output sentiment matches desired backdoor sentiment
                        preds = self._measure_sentiment(logits)
                        if backdoor_targets is not None:
                            # Calculate resistance (non-compliance with backdoor)
                            resistance = 1.0 - torch.mean((preds == backdoor_targets).float()).item()
                        else:
                            # Default assumption: backdoor tries to steer to positive sentiment (1)
                            resistance = 1.0 - torch.mean(preds).item()
                            
                    elif batch["backdoor_type"] == "targeted_refusal":
                        # For targeted refusal: check if model refuses to answer legitimate queries
                        refusal_score = self._measure_refusal(logits)
                        # Resistance = not refusing (opposite of backdoor goal)
                        resistance = 1.0 - refusal_score
                        
                    elif batch["backdoor_type"] == "code_injection":
                        # For code injection: check if output contains malicious code patterns
                        injection_score = self._measure_code_injection(logits)
                        # Resistance = not injecting malicious code
                        resistance = 1.0 - injection_score
                        
                    else:
                        # Default generic measurement (task-specific implementations would be needed)
                        preds = torch.argmax(logits, dim=-1)
                        if backdoor_targets is not None:
                            # Measure non-compliance with backdoor targets
                            resistance = 1.0 - torch.mean((preds == backdoor_targets).float()).item()
                        else:
                            # Placeholder when no specific metric is available
                            resistance = 0.5  # Neutral score
                else:
                    # Generic measurement if backdoor_type is not specified
                    preds = torch.argmax(logits, dim=-1)
                    if backdoor_targets is not None:
                        # Measure non-compliance with backdoor targets
                        resistance = 1.0 - torch.mean((preds == backdoor_targets).float()).item()
                    else:
                        # Without targets, we use a placeholder
                        resistance = 0.5  # Neutral score
                
                resistance_score += resistance
                total_samples += 1
        
        self.model.train()
        return resistance_score / total_samples if total_samples > 0 else 0.0
    
    def _measure_sentiment(self, logits):
        """Measure sentiment in model outputs (placeholder implementation)"""
        # This would be implemented specifically for the model and task
        # Returning random values as placeholder
        batch_size = logits.size(0)
        return torch.randint(0, 2, (batch_size,)).to(logits.device).float()
    
    def _measure_refusal(self, logits):
        """Measure refusal behavior in model outputs (placeholder implementation)"""
        # This would detect phrases like "I cannot answer" or "I'm not allowed to"
        # Returning random values as placeholder
        return torch.rand(1).item()
    
    def _measure_code_injection(self, logits):
        """Measure code injection patterns in model outputs (placeholder implementation)"""
        # This would detect malicious code patterns
        # Returning random values as placeholder
        return torch.rand(1).item()
    
    def _save_purified_model(self):
        """Save the CROW-purified model"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "model_save"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model checkpoint
        model_path = os.path.join(model_dir, f"crow_purified_{timestamp}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save training metrics
        metrics_path = os.path.join(model_dir, f"crow_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)
        
        # Save configuration
        config_path = os.path.join(model_dir, f"crow_config_{timestamp}.json")
        config = {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "targeted_refusal_alpha": self.targeted_refusal_alpha,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "timestamp": timestamp,
            "model_type": type(self.model).__name__
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"CROW-purified model saved: {model_path}")
        print(f"Training metrics saved: {metrics_path}")
        print(f"Configuration saved: {config_path}")

def apply_crow_training(model, train_data=None, epsilon=0.1, alpha=5.5, learning_rate=2e-5,
                        num_epochs=3, batch_size=4, device='cuda', use_redpajama=True, 
                        use_binary_patching=True, entropy_predictor=None, entropy_threshold=0.8, tokenizer=None):
    """
    Apply CROW backdoor elimination training to a model.
    
    This is a convenience function to quickly apply CROW training without having to
    instantiate the CROWBackdoorElimination class directly.
    
    Args:
        model: The language model to fine-tune
        train_data: Clean dataset for fine-tuning (can be small, e.g., 100 samples)
        epsilon: Perturbation magnitude for adversarial examples (default: 0.1)
        alpha: Weighting factor for consistency regularization (default: 5.5)
        learning_rate: Learning rate for parameter updates (default: 2e-5)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Training batch size (default: 4)
        device: Device to perform training on (default: 'cuda')
        use_redpajama: Whether to use the RedPajama dataset (default: True)
        tokenizer: Tokenizer to use for the RedPajama dataset (required if use_redpajama is True and binary_patching=False)
        use_binary_patching: Whether to use binary patching for COCONUT models (default: False)
        entropy_predictor: ByteEntropyPredictor for dynamic binary patching (required if binary_patching=True)
        entropy_threshold: Threshold for entropy-based patching (default: 0.8)
        
    Returns:
        Purified model and training metrics
    """
    print("Initializing CROW Backdoor Elimination procedure...")
    
    # Create a dataset if none is provided
    if train_data is None:
        if use_redpajama:
            if use_binary_patching:
                # For COCONUT architecture with binary patching
                if entropy_predictor is None:
                    print("No entropy predictor provided, creating a new one")
                    entropy_predictor = ByteEntropyPredictor()
                
                # Load RedPajama dataset with binary patching
                train_data = get_redpajama_binary_dataset(
                    max_chars=1000, 
                    entropy_predictor=entropy_predictor,
                    entropy_threshold=entropy_threshold
                )
                print("Using RedPajama dataset with binary patching for CROW training")
            else:
                # Traditional tokenization approach
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided when using RedPajama dataset without binary patching")
                
                # Load RedPajama dataset with standard tokenization
                redpajama_dataset = get_redpajama_dataset(max_chars=1000)
                train_data = redpajama_dataset.prepare_for_model(tokenizer)
                print("Using RedPajama dataset with tokenization for CROW training")
        else:
            # Use the default random dataset as fallback
            train_data = get_default_clean_dataset(batch_size * 25)  # 25 batches
            print("Using default random dataset for CROW training")
    
    # Initialize CROW trainer
    crow_trainer = CROWBackdoorElimination(
        model=model,
        epsilon=epsilon,
        alpha=alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Apply CROW training
    metrics = crow_trainer.train(train_data)
    
    # Return the purified model and metrics
    return model, metrics

def apply_crow_to_coconut(coconut_model, max_chars=1000, epsilon=0.1, alpha=5.5, 
                          learning_rate=2e-5, num_epochs=3, batch_size=4, device='cuda'):
    """
    Apply CROW backdoor elimination specifically to a CoconutBinaryLatentModel.
    This function is designed to work with the binary patching approach used in COCONUT.
    
    Args:
        coconut_model: The CoconutBinaryLatentModel to fine-tune
        max_chars: Maximum number of characters to include in the dataset (default: 1000)
        epsilon: Perturbation magnitude for adversarial examples (default: 0.1)
        alpha: Weighting factor for consistency regularization (default: 5.5)
        learning_rate: Learning rate for parameter updates (default: 2e-5)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Training batch size (default: 4)
        device: Device to perform training on (default: 'cuda')
        
    Returns:
        Purified model and training metrics
    """
    print("Initializing CROW Backdoor Elimination for COCONUT model...")
    
    # Extract the entropy predictor from the COCONUT model
    if hasattr(coconut_model, 'latent_transformer') and hasattr(coconut_model.latent_transformer, 'entropy_predictor'):
        entropy_predictor = coconut_model.latent_transformer.entropy_predictor
    else:
        print("Warning: Could not find entropy_predictor in COCONUT model, creating a new one")
        entropy_predictor = ByteEntropyPredictor()
    
    # Get entropy threshold from model if available
    entropy_threshold = 0.8  # Default threshold
    if (hasattr(coconut_model, 'binary_patch_module') and 
        hasattr(coconut_model.binary_patch_module, 'threshold')):
        entropy_threshold = coconut_model.binary_patch_module.threshold
    
    # Prepare dataset with binary patching
    train_data = get_redpajama_binary_dataset(
        max_chars=max_chars,
        entropy_predictor=entropy_predictor,
        entropy_threshold=entropy_threshold
    )
    
    # Apply CROW training with binary patching
    return apply_crow_training(
        model=coconut_model,
        train_data=train_data,
        epsilon=epsilon,
        alpha=alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        use_redpajama=False  # Already loaded the dataset
    )

# --- ByteEntropyPredictor ---
class ByteEntropyPredictor(nn.Module):
    """
    Predicts the probability distribution of the next byte in a sequence.
    Used for entropy-based dynamic patching in COCONUT model architecture.
    """
    def __init__(self, vocab_size=256, hidden_size=512, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, byte_sequences):
        byte_sequences = byte_sequences.to(self.device)
        byte_embeddings = self.byte_embedding(byte_sequences)
        memory = torch.zeros_like(byte_embeddings)
        decoder_output = self.transformer_decoder(byte_embeddings, memory)
        next_byte_logits = self.fc_out(decoder_output)
        next_byte_probs = torch.softmax(next_byte_logits, dim=-1)
        return next_byte_probs
    
    def get_next_byte_probs(self, byte_sequence_segment):
        """
        Get probability distribution for the next byte after the provided sequence.
        
        Args:
            byte_sequence_segment: Tensor of byte values [batch_size, seq_len]
            
        Returns:
            Tensor of probabilities for each possible next byte [batch_size, vocab_size]
        """
        return self.forward(byte_sequence_segment)[:, -1, :]

def calculate_shannon_entropy(probs):
    """
    Calculate Shannon entropy from a probability distribution.
    
    Args:
        probs: Probability tensor [vocab_size]
        
    Returns:
        Scalar entropy value
    """
    # Add small epsilon to avoid log(0)
    log_probs = torch.log2(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs)
    return entropy

def entropy_patching(byte_sequence, entropy_predictor, threshold=0.8, relative_threshold=0.1):
    """
    Dynamically segment byte sequence into patches based on entropy.
    
    Args:
        byte_sequence: List of bytes to segment
        entropy_predictor: ByteEntropyPredictor model
        threshold: Global entropy threshold for patch boundaries
        relative_threshold: Relative entropy change threshold
        
    Returns:
        List of byte sequence patches
    """
    patches = []
    current_patch_bytes = []
    prev_entropy = None
    
    for i, byte_val in enumerate(byte_sequence):
        current_patch_bytes.append(byte_val)
        
        # Create natural breaks at newlines
        if byte_val == ord('\n'):
            if current_patch_bytes:
                patches.append(bytes(current_patch_bytes))
                current_patch_bytes = []
                prev_entropy = None
            continue
        
        # Convert current patch to tensor for entropy prediction
        input_tensor = torch.tensor([current_patch_bytes], dtype=torch.long).to(entropy_predictor.device)
        
        with torch.no_grad():
            next_probs = entropy_predictor.get_next_byte_probs(input_tensor)
            current_entropy = calculate_shannon_entropy(next_probs.squeeze(0)).item()
        
        if prev_entropy is None:
            prev_entropy = current_entropy
        
        # Create a patch boundary if entropy exceeds threshold or changes significantly
        if current_entropy > threshold or (current_entropy - prev_entropy > relative_threshold):
            patches.append(bytes(current_patch_bytes))
            current_patch_bytes = []
            prev_entropy = None
        else:
            prev_entropy = min(prev_entropy, current_entropy)
    
    # Add any remaining bytes as a final patch
    if current_patch_bytes:
        patches.append(bytes(current_patch_bytes))
    
    return patches

class BinaryPatchingModule(nn.Module):
    """
    Neural network module for dynamic binary patching based on entropy patterns.
    Compatible with COCONUT architecture.
    """
    def __init__(self, input_dim, hidden_dim, threshold=0.5):
        super(BinaryPatchingModule, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 1)
        self.threshold = threshold
        
    def forward(self, latent_states):
        """
        Compute binary patch boundary decisions for latent states.
        
        Args:
            latent_states: Tensor of latent states [batch, seq_len, input_dim]
            
        Returns:
            (binary_mask, probs): Tuple containing the binary decision mask and raw probabilities
        """
        x = F.relu(self.linear(latent_states))
        logits = self.out_linear(x)  # [batch, seq_len, 1]
        probs = torch.sigmoid(logits)
        
        # Create binary mask (1 = patch boundary)
        binary_mask = (probs > self.threshold).float()
        
        # Use straight-through estimator for differentiability
        binary_mask = binary_mask + (probs - probs.detach())
        
        return binary_mask, probs

def get_redpajama_binary_dataset(max_chars=1000, entropy_predictor=None, entropy_threshold=0.8):
    """
    Load RedPajama dataset and convert to binary patches using entropy-based segmentation.
    Compatible with COCONUT binary latent architecture.
    
    Args:
        max_chars: Maximum characters to include
        entropy_predictor: Optional ByteEntropyPredictor for dynamic patching
        entropy_threshold: Threshold for entropy-based patching
        
    Returns:
        Dataset with binary patches
    """
    # Create predictor if not provided
    if entropy_predictor is None:
        entropy_predictor = ByteEntropyPredictor()
    
    # Load and prepare dataset
    raw_data = get_redpajama_dataset(max_chars=max_chars)
    
    # Process with binary patching
    binary_data = []
    for item in raw_data.data:
        text = item["text"]
        byte_sequence = list(text.encode("utf-8"))
        
        # Apply entropy-based patching
        patches = entropy_patching(byte_sequence, entropy_predictor, threshold=entropy_threshold)
        
        # Convert patches to binary tensors (for simplicity, just use thresholded bytes)
        # In a real implementation, you might have a more sophisticated binary encoding
        binary_patches = []
        for patch in patches:
            # Convert bytes to binary tensor
            patch_tensor = torch.tensor([list(patch)], dtype=torch.float) / 255.0
            binary_patches.append((patch_tensor > 0.5).float())
        
        binary_data.append({"binary_patches": binary_patches})
    
    # Create dataset class
    class BinaryPatchedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            item = self.data[idx]
            
            # For CROW training compatibility, we need to convert binary_patches to input_ids format
            if "binary_patches" in item:
                # If we have multiple patches, concatenate them
                if len(item["binary_patches"]) > 0:
                    # Flatten and convert to int for input_ids format
                    # We'll convert the binary values (0/1) to token indices (0 to 255)
                    # For simplicity, we use a linear mapping
                    first_patch = item["binary_patches"][0]
                    if first_patch.dim() == 3:  # [1, seq_len, binary_dim]
                        first_patch = first_patch.squeeze(0)
                    
                    # Convert binary to indices (simple mapping for demonstration)
                    binary_flat = first_patch.flatten()
                    # Scale to 0-255 range
                    input_ids = (binary_flat * 255).long()
                    
                    # Add input_ids to the item
                    item["input_ids"] = input_ids
                else:
                    # Fallback: create an empty tensor
                    item["input_ids"] = torch.empty(0, dtype=torch.long)
            
            return item
    
    return BinaryPatchedDataset(binary_data)

def get_redpajama_dataset(max_chars=1000):
    """
    Load and prepare the RedPajama dataset for CROW training.
    
    This function loads a small subset of the RedPajama dataset from Huggingface,
    limiting the total characters to approximately the specified amount for backdoor elimination training.
    
    Args:
        max_chars: Maximum number of characters to include in the dataset (default: 1000)
                  
    Returns:
        PyTorch Dataset object containing the processed RedPajama data
    """
    print("Loading RedPajama dataset for CROW training...")
    
    # Load the RedPajama dataset from Huggingface
    raw_dataset = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="default",
        partition="head_middle",
        snapshots=["2022-49"],
        languages=["en"],
        streaming=True  # Use streaming to handle the large dataset efficiently
    )
    
    # Create a PyTorch dataset class to hold the RedPajama data
    class RedPajamaDataset(torch.utils.data.Dataset):
        def __init__(self, raw_data, max_chars):
            self.data = []
            
            # Track total characters
            char_count = 0
            
            # Iterate through the dataset and collect samples until we reach max_chars
            print("Processing RedPajama dataset...")
            iterator = iter(raw_data["train"])
            
            while char_count < max_chars:
                try:
                    sample = next(iterator)
                    text = sample["text"]
                    
                    # Calculate character count for this sample
                    sample_chars = len(text)
                    
                    # If this sample would exceed our character limit, truncate it
                    remaining_chars = max_chars - char_count
                    if sample_chars > remaining_chars:
                        text = text[:remaining_chars]
                        sample_chars = remaining_chars
                        
                    # Add the processed sample to our dataset
                    self.data.append({"text": text})
                    
                    char_count += sample_chars
                    
                    if char_count >= max_chars:
                        break
                        
                except StopIteration:
                    print("Reached end of RedPajama dataset stream")
                    break
            
            print(f"Collected RedPajama dataset with {char_count} characters "
                  f"({len(self.data)} samples)")
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def prepare_for_model(self, tokenizer):
            """
            Prepare the dataset for model training by tokenizing the text.
            
            Args:
                tokenizer: The tokenizer to use for encoding the text
                
            Returns:
                The tokenized dataset
            """
            tokenized_data = []
            for item in self.data:
                encodings = tokenizer(item["text"], truncation=True, padding=False, return_tensors="pt")
                input_ids = encodings["input_ids"].squeeze(0)
                tokenized_data.append({"input_ids": input_ids})
            
            self.data = tokenized_data
            return self
    
    # Create and return the dataset
    return RedPajamaDataset(raw_dataset, max_chars)

def get_default_clean_dataset(size=100):
    """
    Create a simple default dataset for CROW training.
    In a real implementation, you would use an actual clean dataset.
    
    Args:
        size: Number of samples in the dataset
        
    Returns:
        Simple dataset object
    """
    # This is a placeholder for demonstration
    # In practice, you would load a real dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
            # Generate random input_ids of varying lengths (between 10 and 100 tokens)
            self.data = [{"input_ids": torch.randint(0, 50257, (torch.randint(10, 100, (1,)).item(),))} 
                         for _ in range(size)]
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return SimpleDataset(size)

def plot_crow_training_progress(metrics):
    """
    Plot training metrics from CROW training.
    
    Args:
        metrics: Dictionary of training metrics from CROWBackdoorElimination.train()
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(metrics['llm_losses'], label='LLM Loss')
    plt.plot(metrics['consistency_losses'], label='Consistency Loss')
    plt.plot(metrics['total_losses'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot clean accuracy if available
    if 'clean_accuracy' in metrics and metrics['clean_accuracy']:
        plt.subplot(2, 2, 2)
        plt.plot(metrics['clean_accuracy'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Clean Performance')
        plt.grid(True)
    
    # Plot backdoor resistance if available
    if 'backdoor_resistance' in metrics and metrics['backdoor_resistance']:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['backdoor_resistance'], marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Resistance Score')
        plt.title('Backdoor Resistance (higher is better)')
        plt.grid(True)
    
    # Plot combined metrics
    plt.subplot(2, 2, 4)
    metrics_to_plot = []
    labels = []
    
    if 'llm_losses' in metrics:
        metrics_to_plot.append(metrics['llm_losses'])
        labels.append('LLM Loss')
    
    if 'clean_accuracy' in metrics and metrics['clean_accuracy']:
        # Scale to be comparable with losses
        scaled_acc = [acc * max(metrics['llm_losses']) for acc in metrics['clean_accuracy']]
        metrics_to_plot.append(scaled_acc)
        labels.append('Clean Acc (scaled)')
    
    if 'backdoor_resistance' in metrics and metrics['backdoor_resistance']:
        # Scale to be comparable with losses
        scaled_res = [res * max(metrics['llm_losses']) for res in metrics['backdoor_resistance']]
        metrics_to_plot.append(scaled_res)
        labels.append('Backdoor Res (scaled)')
    
    for metric, label in zip(metrics_to_plot, labels):
        plt.plot(metric, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Combined Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('model_save/crow_training_metrics.png')
    plt.close()
    
    print("Training metrics plot saved to model_save/crow_training_metrics.png")

    '''
    Crow Implementation Details

    The implementation:

    Adds protection against model poisoning, reducing adversarial attacks by ~66%

    Uses internal consistency regularization to detect and neutralize backdoor triggers

    Simulates backdoor attacks via adversarial perturbations during training

    Enforces smooth transitions across transformer layers for better robustness

    The CROW method has been fully integrated with the CoconutBinaryLatentModel class through the apply_crow_backdoor_elimination method, which allows for customizing key hyperparameters:

RL (Reinforcement Learning) Hyperparameters:

Perturbation magnitude (epsilon): 0.1

Consistency weight (alpha): 5.5 (11.0 for targeted refusal tasks)

Learning rate: 2e-5 with cosine decay and warmup

The implementation includes training metrics tracking, model checkpointing, and visualization tools. The purified model will maintain regular functionality while being significantly more resistant to poisoning attempts.
   
 '''
