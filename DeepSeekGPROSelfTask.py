import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import datetime
import logging
import collections
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpro_self_task.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepSeekGPRO")

class ThinkingChainsDataset(Dataset):
    """
    Dataset for pre-training the model on thinking chains examples.
    
    This is a placeholder that will be filled with actual data for training
    before the GPRO self-learning phase.
    """
    def __init__(self, data_path: str = None):
        """
        Initialize the thinking chains dataset.
        
        Args:
            data_path: Path to the dataset file (JSON format expected).
                       If None, uses an empty list for placeholder purposes.
        """
        self.examples = []
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
            logger.info(f"Loaded {len(self.examples)} thinking chain examples from {data_path}")
        else:
            logger.warning("No data path provided or file not found. Using empty dataset as placeholder.")
    
    def load_data(self, data_path: str):
        """Load thinking chain examples from a JSON file."""
        try:
            with open(data_path, 'r') as f:
                self.examples = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            self.examples = []
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class GroupRelativePolicyOptimization:
    """
    Implementation of the Group Relative Policy Optimization (GRPO) algorithm
    for the self-task component of the moral algorithm.
    
    This implements the DeepSeek R1 GPRO approach, which samples groups of outputs
    for each question and optimizes using relative advantage within the group.
    """
    def __init__(
        self,
        model,
        optimizer,
        group_size: int = 8,               # G: Number of outputs sampled per question
        epsilon: float = 0.2,              # ε: Clipping parameter for probability ratio
        beta: float = 0.01,                # β: KL regularization coefficient
        learning_rate: float = 2e-5,       # Learning rate for the optimizer
        max_grad_norm: float = 1.0,        # Maximum gradient norm for clipping
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reference_model = None,            # π_ref: Reference policy model (if None, uses a copy of the initial model)
        use_advantage_normalization: bool = True,  # Whether to normalize advantages
    ):
        """
        Initialize the GPRO algorithm.
        
        Args:
            model: The COCONUT model to train
            optimizer: Optimizer for updating model parameters
            group_size: Number of outputs to sample per question (G)
            epsilon: Clipping parameter for probability ratio (ε)
            beta: KL regularization coefficient (β)
            learning_rate: Learning rate for the optimizer
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run training on
            reference_model: Reference policy model (π_ref). If None, a copy of the initial model will be used.
            use_advantage_normalization: Whether to normalize advantages within groups
        """
        self.model = model
        self.optimizer = optimizer
        self.group_size = group_size
        self.epsilon = epsilon
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_advantage_normalization = use_advantage_normalization
        
        # Initialize reference model (π_ref)
        if reference_model is None:
            # Create a copy of the initial model
            self.reference_model = self._create_reference_model_copy(model)
            logger.info("Created reference model as a copy of the initial model")
        else:
            self.reference_model = reference_model
            logger.info("Using provided reference model")
        
        # Set reference model to evaluation mode (no gradient tracking needed)
        self.reference_model.eval()
        
        # Track old policy parameters (will be updated before each training iteration)
        self.old_policy = self._create_reference_model_copy(model)
        self.old_policy.eval()
        
        # Initialize KL divergence calculation
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Metrics tracking
        self.metrics = {
            'policy_loss': [],
            'kl_loss': [],
            'total_loss': [],
            'mean_reward': [],
            'mean_advantage': [],
            'norm_advantage': []
        }

    def _create_reference_model_copy(self, model):
        """
        Create a copy of the model for use as reference or old policy.
        
        Args:
            model: The model to copy
            
        Returns:
            A copy of the model with the same parameters but no gradients
        """
        # Create a new instance (this depends on your model's structure)
        # If your model has a clone() method, use that, otherwise use state_dict
        if hasattr(model, 'clone'):
            model_copy = model.clone()
        else:
            # This is a simplified approach - adapt based on your specific model
            try:
                # Try accessing _args and _kwargs, but might not be available
                model_copy = type(model)(*getattr(model, '_args', []), **getattr(model, '_kwargs', {}))
            except (AttributeError, TypeError):
                # Fallback: Create a new instance with the same class
                model_copy = type(model)()
            
            # Copy state dict
            model_copy.load_state_dict(model.state_dict())
        
        # Make sure the copy doesn't track gradients
        for param in model_copy.parameters():
            param.requires_grad = False
            
        return model_copy.to(self.device)

    def update_old_policy(self):
        """Update the old policy with the current model parameters."""
        self.old_policy.load_state_dict(self.model.state_dict())
        logger.info("Updated old policy with current model parameters")

    def compute_rewards(
        self,
        outputs: List[List[dict]],
        queries: List[str],
        reward_fn: Callable[[str, str], float]
    ) -> torch.Tensor:
        """
        Compute rewards for each output in the group using a reward function.
        
        Args:
            outputs: List of lists of model outputs (each inner list has group_size elements)
            queries: List of input queries corresponding to outputs
            reward_fn: Function that computes reward given query and output
            
        Returns:
            Tensor of rewards for each output [batch_size, group_size]
        """
        rewards = []
        
        for output_group, query in zip(outputs, queries):
            group_rewards = []
            
            for output in output_group:
                # Get the output text from the output dictionary
                output_text = output.get('text', '')
                
                # Compute reward using the provided reward function
                reward = reward_fn(query, output_text)
                group_rewards.append(reward)
            
            rewards.append(group_rewards)
            
        return torch.tensor(rewards, device=self.device)

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages for each output in the group.
        
        Follows the formula:
        A_i = (r_i - mean({r_1, r_2, …, r_G})) / std({r_1, r_2, …, r_G})
        
        Args:
            rewards: Tensor of rewards for each output in the group [batch_size, group_size]
            
        Returns:
            Tensor of advantages for each output [batch_size, group_size]
        """
        if self.use_advantage_normalization:
            # Compute mean and standard deviation for each group
            group_means = rewards.mean(dim=1, keepdim=True)  # [batch_size, 1]
            group_stds = rewards.std(dim=1, keepdim=True)    # [batch_size, 1]
            
            # Add small epsilon to avoid division by zero
            group_stds = torch.clamp(group_stds, min=1e-8)
            
            # Compute advantages using normalization
            advantages = (rewards - group_means) / group_stds  # [batch_size, group_size]
        else:
            # If not using normalization, advantages are just the rewards
            advantages = rewards
            
        return advantages

    def compute_kl_divergence(
        self,
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        Follows the formula:
        D_KL(π_θ || π_ref) = (π_ref(o_i|q) / π_θ(o_i|q)) - log(π_ref(o_i|q) / π_θ(o_i|q)) - 1
        
        Args:
            current_log_probs: Log probabilities from current policy [batch_size, group_size]
            reference_log_probs: Log probabilities from reference policy [batch_size, group_size]
            
        Returns:
            KL divergence loss
        """
        # Compute KL divergence using the formula directly
        # Convert log probs to probs
        current_probs = torch.exp(current_log_probs)
        reference_probs = torch.exp(reference_log_probs)
        
        # Compute ratio of probs
        ratio = reference_probs / (current_probs + 1e-10)
        
        # Compute log of ratio
        log_ratio = reference_log_probs - current_log_probs
        
        # D_KL(π_ref || π_θ) = (π_ref / π_θ) - log(π_ref / π_θ) - 1
        kl_div = ratio - log_ratio - 1.0
        
        # Average over batch and group dimensions
        return kl_div.mean()

    def compute_policy_loss(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the policy loss component of GPRO.
        
        Follows the formula:
        min((π_θ(o_i|q) / π_{θ_old}(o_i|q)) * A_i, clip(π_θ(o_i|q) / π_{θ_old}(o_i|q), 1-ε, 1+ε) * A_i)
        
        Args:
            current_log_probs: Log probabilities from current policy [batch_size, group_size]
            old_log_probs: Log probabilities from old policy [batch_size, group_size]
            advantages: Advantages for each output [batch_size, group_size]
            
        Returns:
            Policy loss
        """
        # Compute probability ratio: π_θ(o_i|q) / π_{θ_old}(o_i|q)
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Take the minimum to implement the clipped objective
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        return policy_loss

    def train_step(
        self,
        queries: List[str],
        outputs_data: List[List[dict]],
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step using the GPRO algorithm.
        
        Args:
            queries: List of input queries
            outputs_data: List of lists of model outputs (each inner list has group_size elements)
            rewards: Tensor of rewards for each output [batch_size, group_size]
            
        Returns:
            Dictionary of metrics for this training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = len(queries)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Track current metrics
        step_metrics = {
            'policy_loss': 0.0,
            'kl_loss': 0.0,
            'total_loss': 0.0,
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'norm_advantage': advantages.norm().item()
        }
        
        # Compute log probabilities for each output with current model
        current_log_probs = []
        old_log_probs = []
        reference_log_probs = []
        
        for b in range(batch_size):
            query = queries[b]
            output_group = outputs_data[b]
            
            batch_current_log_probs = []
            batch_old_log_probs = []
            batch_reference_log_probs = []
            
            for output in output_group:
                # Get output text and token IDs
                output_text = output.get('text', '')
                output_tokens = output.get('token_ids', [])
                
                # Get log probabilities from current model
                with torch.set_grad_enabled(True):
                    current_log_prob = self.model.get_log_prob(query, output_text, output_tokens)
                    batch_current_log_probs.append(current_log_prob)
                
                # Get log probabilities from old policy (no grad)
                with torch.no_grad():
                    old_log_prob = self.old_policy.get_log_prob(query, output_text, output_tokens)
                    batch_old_log_probs.append(old_log_prob)
                    
                    # Get log probabilities from reference model (no grad)
                    ref_log_prob = self.reference_model.get_log_prob(query, output_text, output_tokens)
                    batch_reference_log_probs.append(ref_log_prob)
            
            # Stack log probabilities for this batch
            current_log_probs.append(torch.stack(batch_current_log_probs))
            old_log_probs.append(torch.stack(batch_old_log_probs))
            reference_log_probs.append(torch.stack(batch_reference_log_probs))
        
        # Stack all batch results
        current_log_probs = torch.stack(current_log_probs)  # [batch_size, group_size]
        old_log_probs = torch.stack(old_log_probs)          # [batch_size, group_size]
        reference_log_probs = torch.stack(reference_log_probs)  # [batch_size, group_size]
        
        # Compute policy loss
        policy_loss = self.compute_policy_loss(current_log_probs, old_log_probs, advantages)
        step_metrics['policy_loss'] = policy_loss.item()
        
        # Compute KL divergence to reference model
        kl_loss = self.compute_kl_divergence(current_log_probs, reference_log_probs)
        step_metrics['kl_loss'] = kl_loss.item()
        
        # Compute total loss
        total_loss = policy_loss + self.beta * kl_loss
        step_metrics['total_loss'] = total_loss.item()
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Update metrics history
        for key, value in step_metrics.items():
            self.metrics[key].append(value)
        
        return step_metrics

    def save_metrics(self, filepath: str = "gpro_metrics.json"):
        """
        Save training metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics file
        """
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved training metrics to {filepath}")

    def log_metrics(self, step_metrics: Dict[str, float], step: int):
        """
        Log current training metrics.
        
        Args:
            step_metrics: Dictionary of metrics for the current step
            step: Current training step
        """
        logger.info(f"Step {step} - "
                   f"Loss: {step_metrics['total_loss']:.4f} "
                   f"(Policy: {step_metrics['policy_loss']:.4f}, "
                   f"KL: {step_metrics['kl_loss']:.4f}) | "
                   f"Mean Reward: {step_metrics['mean_reward']:.4f} | "
                   f"Mean Advantage: {step_metrics['mean_advantage']:.4f}")


class GPROSelfTaskTrainer:
    """
    Trainer for the self-task component using the GPRO algorithm.
    
    This trains the COCONUT model to improve its reasoning capabilities
    through self-learning on synthetic data generated from its own outputs.
    """
    def __init__(
        self,
        model,
        reward_calculator,
        group_size: int = 8,
        epsilon: float = 0.2,
        beta: float = 0.01,
        learning_rate: float = 2e-5,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "gpro_checkpoints",
        reference_model = None,
        reward_fn = None,
    ):
        """
        Initialize the GPRO Self-Task Trainer.
        
        Args:
            model: The COCONUT model to train
            reward_calculator: Function or class that calculates rewards for model outputs
            group_size: Number of outputs to sample per question (G)
            epsilon: Clipping parameter for probability ratio (ε)
            beta: KL regularization coefficient (β)
            learning_rate: Learning rate for the optimizer
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run training on
            checkpoint_dir: Directory to save model checkpoints
            reference_model: Reference policy model (π_ref)
            reward_fn: Optional custom reward function
        """
        self.model = model
        self.reward_calculator = reward_calculator
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.reward_fn = reward_fn or self._default_reward_fn
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Initialize GPRO algorithm
        self.gpro = GroupRelativePolicyOptimization(
            model=model,
            optimizer=self.optimizer,
            group_size=group_size,
            epsilon=epsilon,
            beta=beta,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            device=device,
            reference_model=reference_model
        )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training stats
        self.training_stats = {
            'iterations': 0,
            'total_queries': 0,
            'start_time': None,
            'end_time': None,
            'best_reward': float('-inf'),
            'best_model_path': None
        }
    
    def _default_reward_fn(self, query: str, output: str) -> float:
        """
        Default reward function that uses the reward calculator.
        
        Args:
            query: Input query
            output: Model output
            
        Returns:
            Reward value
        """
        # This implementation assumes reward_calculator can compute a reward
        # given a query and output. You'll need to adapt this to your specific
        # reward calculator's interface.
        model_output = self._convert_to_model_output_format(output)
        return self.reward_calculator.calculate_reward(query=query, response=output, model_output=model_output)
    
    def _convert_to_model_output_format(self, output: str):
        """
        Convert raw output string to the format expected by the reward calculator.
        
        Args:
            output: Raw output string
            
        Returns:
            Output in the format expected by the reward calculator
        """
        # This is an example implementation - adapt to your model output format
        return {
            'text': output,
            'shape': torch.Size([1, len(output)]),  # Approximate shape based on output length
            'logits': None,  # In a real implementation, compute these
            'token_ids': None  # In a real implementation, compute these
        }
    
    def sample_outputs(self, query: str, n_samples: int) -> List[dict]:
        """
        Sample multiple outputs from the model for a single query.
        
        Args:
            query: Input query
            n_samples: Number of outputs to sample
            
        Returns:
            List of sampled outputs with necessary metadata
        """
        outputs = []
        
        for _ in range(n_samples):
            # Sample an output from the model
            # This implementation depends on your specific model interface
            # In a real implementation, you would call model.generate() or similar
            output = self.model.sample(query)
            
            # Add to outputs list
            outputs.append(output)
        
        return outputs
    
    def generate_synthetic_data(
        self,
        queries: List[str],
        group_size: int
    ) -> Tuple[List[str], List[List[dict]]]:
        """
        Generate synthetic data by sampling multiple outputs for each query.
        
        Args:
            queries: List of input queries
            group_size: Number of outputs to sample per query
            
        Returns:
            Tuple of (queries, grouped_outputs)
        """
        grouped_outputs = []
        
        for query in queries:
            # Sample group_size outputs for this query
            group_outputs = self.sample_outputs(query, group_size)
            grouped_outputs.append(group_outputs)
        
        return queries, grouped_outputs
    
    def train_iteration(
        self,
        queries: List[str],
        grouped_outputs: List[List[dict]] = None,
        custom_rewards: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Perform a single training iteration using the GPRO algorithm.
        
        Args:
            queries: List of input queries
            grouped_outputs: Optional pre-generated grouped outputs
            custom_rewards: Optional pre-computed rewards
            
        Returns:
            Dictionary of metrics for this training iteration
        """
        # If no grouped outputs provided, generate them
        if grouped_outputs is None:
            queries, grouped_outputs = self.generate_synthetic_data(
                queries, self.gpro.group_size
            )
        
        # If no custom rewards provided, compute them using the reward function
        if custom_rewards is None:
            rewards = self.gpro.compute_rewards(grouped_outputs, queries, self.reward_fn)
        else:
            rewards = custom_rewards
        
        # Update old policy before training step
        self.gpro.update_old_policy()
        
        # Perform GPRO training step
        metrics = self.gpro.train_step(queries, grouped_outputs, rewards)
        
        # Update training stats
        self.training_stats['iterations'] += 1
        self.training_stats['total_queries'] += len(queries)
        
        # Check if this is a new best model
        if metrics['mean_reward'] > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = metrics['mean_reward']
            
            # Save best model checkpoint
            best_model_path = os.path.join(
                self.checkpoint_dir,
                f"best_model_iter_{self.training_stats['iterations']}.pt"
            )
            self.save_checkpoint(best_model_path)
            self.training_stats['best_model_path'] = best_model_path
        
        return metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        eval_freq: int = 10,
        save_freq: int = 100,
        log_freq: int = 10
    ):
        """
        Train the model using the GPRO algorithm.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            eval_freq: Frequency of evaluation (in iterations)
            save_freq: Frequency of saving checkpoints (in iterations)
            log_freq: Frequency of logging metrics (in iterations)
        """
        # Record start time
        self.training_stats['start_time'] = time.time()
        
        logger.info(f"Starting GPRO training for {num_epochs} epochs")
        
        total_iterations = 0
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                
                for batch_idx, batch in enumerate(dataloader):
                    # Extract queries from batch
                    # Adapt this based on your specific batch format
                    queries = batch['queries'] if isinstance(batch, dict) else batch
                    
                    # Perform training iteration
                    metrics = self.train_iteration(queries)
                    total_iterations += 1
                    
                    # Log metrics
                    if total_iterations % log_freq == 0:
                        self.gpro.log_metrics(metrics, total_iterations)
                    
                    # Evaluate model
                    if total_iterations % eval_freq == 0:
                        eval_metrics = self.evaluate()
                        logger.info(f"Evaluation metrics at iteration {total_iterations}: {eval_metrics}")
                    
                    # Save checkpoint
                    if total_iterations % save_freq == 0:
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir,
                            f"model_iter_{total_iterations}.pt"
                        )
                        self.save_checkpoint(checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Play sound to indicate checkpoint saved
                        self._play_sound()
            
            # Record end time
            self.training_stats['end_time'] = time.time()
            
            # Save final model and metrics
            final_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                "model_final.pt"
            )
            self.save_checkpoint(final_checkpoint_path)
            
            self.gpro.save_metrics(os.path.join(
                self.checkpoint_dir,
                "gpro_metrics.json"
            ))
            
            self.save_training_stats()
            
            logger.info(f"Training completed. Best model saved at: {self.training_stats['best_model_path']}")
            
            # Play sound to indicate training completion
            self._play_sound()
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            # Try to save checkpoint on error
            try:
                emergency_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    "model_emergency.pt"
                )
                self.save_checkpoint(emergency_checkpoint_path)
                logger.info(f"Saved emergency checkpoint to {emergency_checkpoint_path}")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
            
            raise e
    
    def _play_sound(self, sound_file: str = "Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav"):
        """Play a sound to indicate completion of a training phase."""
        try:
            import platform
            import subprocess
            
            if platform.system() == "Linux":
                subprocess.run(["aplay", sound_file])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["afplay", sound_file])
            elif platform.system() == "Windows":
                import winsound
                winsound.PlaySound(sound_file, winsound.SND_FILENAME)
            logger.info(f"Sound played: {sound_file}")
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
    
    def evaluate(self, eval_queries: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model on a set of queries.
        
        Args:
            eval_queries: List of evaluation queries. If None, uses a default set.
            
        Returns:
            Dictionary of evaluation metrics
        """
        # If no eval queries provided, use defaults
        if eval_queries is None:
            eval_queries = self._get_default_eval_queries()
        
        self.model.eval()
        
        total_reward = 0.0
        all_rewards = []
        
        with torch.no_grad():
            for query in eval_queries:
                # Generate one output for evaluation
                # This implementation depends on your model interface
                output = self.model.generate(query)
                
                # Calculate reward
                if isinstance(output, dict) and 'text' in output:
                    output_text = output['text']
                else:
                    output_text = str(output)
                    
                reward = self.reward_fn(query, output_text)
                
                total_reward += reward
                all_rewards.append(reward)
        
        # Calculate metrics
        mean_reward = total_reward / len(eval_queries) if eval_queries else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min(all_rewards) if all_rewards else 0,
            'max_reward': max(all_rewards) if all_rewards else 0
        }
    
    def _get_default_eval_queries(self) -> List[str]:
        """
        Get a default set of evaluation queries.
        
        Returns:
            List of default evaluation queries
        """
        # Default evaluation queries focusing on reasoning tasks
        return [
            "Explain the concept of reinforcement learning.",
            "What are the key components of a neural network?",
            "How would you solve the traveling salesman problem?",
            "What is the relationship between artificial intelligence and machine learning?",
            "Describe an algorithm to find the maximum sum subarray.",
            "Explain how transformers work in natural language processing.",
            "What are the ethical considerations of deploying AI systems?",
            "How would you implement a system to detect emotions in text?",
            "Explain the concept of information entropy.",
            "How would you design a recommendation system from scratch?"
        ]
    
    def save_checkpoint(self, filepath: str):
        """
        Save a model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gpro_metrics': self.gpro.metrics,
            'training_stats': self.training_stats,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.gpro.metrics = checkpoint['gpro_metrics']
        self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def save_training_stats(self, filepath: str = None):
        """
        Save training statistics to a JSON file.
        
        Args:
            filepath: Path to save the stats file. If None, uses default path.
        """
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "training_stats.json")
        
        # Calculate training duration
        if self.training_stats['start_time'] and self.training_stats['end_time']:
            duration = self.training_stats['end_time'] - self.training_stats['start_time']
            # Format as HH:MM:SS
            duration_str = str(datetime.timedelta(seconds=int(duration)))
            self.training_stats['duration'] = duration_str
        
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
            
        logger.info(f"Saved training stats to {filepath}")


class EnvironmentalRewardCalculator:
    """
    Environmental reward calculator based on the provided environmental reward
    function in the specifications.
    
    This implements the reward calculation for tasks, reasoning quality,
    tag usage, and other aspects from the moral algorithm's environmental
    reward component.
    """
    def __init__(
        self,
        output_length_weight: float = 0.01,
        log_prob_weight: float = 0.1,
        cot_complexity_weight: float = 0.05,
        eos_tag_reward: float = 1.0,
        output_tag_reward: float = 1.0,
        tool_tag_reward: float = 1.0,
        audio_tag_reward: float = 1.0,
        language_reward: float = 1.0,
        perception_reward: float = 1.0,
        accuracy_reward_weight: float = 1.0,
        repetition_penalty_weight: float = 0.1,
        brevity_reward_weight: float = 0.5,
        latent_space_efficiency_weight: float = 1.0,  # Weight for latent space efficiency reward
        latent_penalty_weight: float = 0.1,           # Penalty weight for each additional latent space
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the environmental reward calculator.
        
        Args:
            output_length_weight: Weight for output length reward
            log_prob_weight: Weight for log probability reward
            cot_complexity_weight: Weight for chain-of-thought complexity reward
            eos_tag_reward: Reward for correct EOS tag usage
            output_tag_reward: Reward for correct output tag usage
            tool_tag_reward: Reward for correct tool tag usage
            audio_tag_reward: Reward for correct audio tag usage
            language_reward: Reward for using the same language as user
            perception_reward: Reward for perspective-taking
            accuracy_reward_weight: Weight for accuracy reward
            repetition_penalty_weight: Weight for repetition penalty
            brevity_reward_weight: Weight for brevity reward
            device: Device to run calculations on
        """
        self.output_length_weight = output_length_weight
        self.log_prob_weight = log_prob_weight
        self.cot_complexity_weight = cot_complexity_weight
        self.eos_tag_reward = eos_tag_reward
        self.output_tag_reward = output_tag_reward
        self.tool_tag_reward = tool_tag_reward
        self.audio_tag_reward = audio_tag_reward
        self.language_reward = language_reward
        self.perception_reward = perception_reward
        self.accuracy_reward_weight = accuracy_reward_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.brevity_reward_weight = brevity_reward_weight
        self.latent_space_efficiency_weight = latent_space_efficiency_weight
        self.latent_penalty_weight = latent_penalty_weight
        self.device = device
        
    def calculate_reward(
        self,
        model_output=None,
        labels=None,
        average_log_prob=None,
        metacognitive_output=None,
        generated_token_ids=None,
        logits=None,
        state=None,
        action=None,
        next_state=None,
        reasoning_quality_reward=0.0,
        query=None,
        response=None,
        correct_response_checker=None,
        loss=None,
        binary_mask=None
    ):
        """
        Calculate environmental reward based on model output and various factors.
        
        This implements the environmental reward function from the specifications,
        calculating rewards for output length, log probability, chain-of-thought
        complexity, tag usage, language matching, and perception.
        
        Args:
            model_output: Output from the model
            labels: Target labels
            average_log_prob: Average log probability of generated tokens
            metacognitive_output: Output from metacognitive reflection
            generated_token_ids: IDs of generated tokens
            logits: Model logits
            state: Current state
            action: Current action
            next_state: Next state
            reasoning_quality_reward: Additional reward for reasoning quality
            query: Input query
            response: Generated response
            correct_response_checker: Function to check response correctness
            
        Returns:
            Total environmental reward
        """
        import collections
        
        reward = 0.0
        
        # 1. Output Length Reward
        if model_output is not None and hasattr(model_output, 'shape'):
            output_length_reward = model_output.shape[1] * self.output_length_weight
            reward += output_length_reward
        
        # 2. Log Probability Reward
        log_prob_reward = 0.0
        if generated_token_ids is not None and logits is not None:
            # Convert token IDs list to tensor
            device = logits.device if hasattr(logits, 'device') else self.device
            generated_token_ids_tensor = torch.tensor(
                generated_token_ids, device=device
            ).unsqueeze(0)  # Assuming batch_size=1
            
            # Get log probabilities for the generated tokens
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            
            # Gather log probabilities of the generated tokens
            generated_log_probs = torch.gather(
                log_probs.view(-1, log_probs.size(-1)),
                1,
                generated_token_ids_tensor.view(-1, 1)
            )
            
            local_average_log_prob = generated_log_probs.mean()
            log_prob_reward = local_average_log_prob * self.log_prob_weight
            reward += log_prob_reward
        elif average_log_prob is not None:
            log_prob_reward = average_log_prob * self.log_prob_weight
            reward += log_prob_reward
        
        # 3. CoT Complexity Reward
        cot_complexity_reward = 0.0
        if metacognitive_output is not None and 'reasoning_quality_score' in metacognitive_output:
            cot_complexity_reward = metacognitive_output['reasoning_quality_score'] * self.cot_complexity_weight
            reward += cot_complexity_reward
        
        # 4. Task-Specific Reasoning Metrics
        # Additional reasoning reward added directly from parameter
        reward += reasoning_quality_reward
        
        # 5. Tag Usage Rewards
        if response is not None:
            # Check for tag presence
            eos_tag_present = '<eos>' in response and '</eos>' in response
            output_tag_present = '<output>' in response and '</output>' in response
            tool_tag_present = '<tool>' in response and '</tool>' in response
            audio_tag_present = '<audio>' in response and '</audio>' in response
            
            tag_reward = 0.0
            # EOS tag rewards/penalties
            if eos_tag_present:
                tag_reward += self.eos_tag_reward
            else:
                tag_reward -= self.eos_tag_reward  # Penalty for missing EOS tag
            
            # Output tag rewards/penalties
            if output_tag_present:
                tag_reward += self.output_tag_reward
            else:
                tag_reward -= self.output_tag_reward  # Penalty for missing output tag
            
            # Tool tag rewards (only if needed)
            if query is not None and 'tool' in query.lower() and tool_tag_present:
                tag_reward += self.tool_tag_reward
            
            # Audio tag rewards (only if needed)
            if query is not None and 'audio' in query.lower() and audio_tag_present:
                tag_reward += self.audio_tag_reward
            
            reward += tag_reward
        
        # 6. Same Language Reward
        # This would require language detection, but for simplicity:
        # If response is in English when query is in English (default case)
        # or if both are in a non-English language, reward is positive
        reward += self.language_reward
        
        # 7. Perception Reward
        # Reward for perspective-taking (simplified implementation)
        reward += self.perception_reward
        
        # 8. Accuracy Reward based on Ground Truth
        if query is not None and response is not None and correct_response_checker is not None:
            if correct_response_checker(query, response):
                accuracy_reward = 10.0 * self.accuracy_reward_weight
                reward += accuracy_reward
            else:
                accuracy_penalty = -2.0 * self.accuracy_reward_weight
                reward += accuracy_penalty
        
        # 9. Repetition Penalty
        if response is not None:
            words = response.lower().split()  # Simple tokenization
            word_counts = collections.Counter(words)
            repeated_word_penalty = 0
            for word, count in word_counts.items():
                if count > 1 and len(word) > 2:  # Penalize repeated words longer than 2 chars
                    repeated_word_penalty += (count - 1)
            
            repetition_penalty = -repeated_word_penalty * self.repetition_penalty_weight
            reward += repetition_penalty
        
        # 10. Brevity Reward
        if response is not None:
            response_length_bytes = len(response.encode('utf-8'))
            brevity_reward = max(0, 5.0 - (response_length_bytes / 20.0)) * self.brevity_reward_weight
            reward += brevity_reward
        
        # ---- Flame Guard Meta - Inspired Deep Research Reward -----
        Fact_based_Check_reward = 1.0     #This rewards the llm for successfully checking and verifying whether the user is asking for a non-fact-based response (fictional, only opinion-based, etc.)
        knowledge_base_Check_reward = 1.0 #This encourages the llm to check its own knowledge-base first if the question from the user is fact-based to see if the answer is there.
                                          #If this knowledge is sufficient and completly answers the user's questions, then the llm may continue without searching. However, if the llm's confidence on its answer is still low (lower than .5), then it may search the web as well.
        Search_Check_reward = 1.0 #This rewards the llm for using a tool-call or deep research-based tools for verifying its own response for correctness from a site as
                                  #well before presenting its final answer in the output. The llm will be required to search for 3 sites from verified sources
                                  # to verify its answer. The sites are as follows: Wikipedia (General Knowledge), Mayo Clinic (medical knowledge), Internet Archive/Wayback Machine (Old website and public library books),
                                  # https://arxiv.org/ (Science Reports and Machine Learning Reports that are peer reviewed by researchers for legitimacy), public library listings (https://www.usa.gov/libraries-and-archives),
                                  # Digital Library of America (Historical Online Resources and Books) https://dp.la/browse-by-topic, oxford dictionary (verified thesaurus and word resource): https://www.oed.com/?tl=true,
                                  # Unity (Game Engine for making video games; contains information about how to use the engine) (https://unity.com/),
                                  # PBS News (News Station about recent and past events funded by the state government): https://www.pbs.org/newshour/,
                                  # Unreal Engine (Game Engine for creating video games; contains information about how to use the engine): https://www.unrealengine.com/en-US,
                                  # Stock market data (CNN News network site which has up to date stock market information) (https://www.cnn.com/markets),
                                  #before presenting the output. The llm will also need to specify if, after this information search, if it is confident or not very confident
                                  #in its answer. The URL and site search will also have the URLs captured in logs from the tool call that the llm used
                                  # and presented as links below the LLM's output so that the user may self-verify the information.

        Verify_check_reward = 1.0 #Rewards the LLM for specifying what its correct confidence level is for how true it believes its answer is.

        # Add flame guard rewards if relevant signals are present in the response
        if response is not None and query is not None:
            # Check for fact verification signals in response
            if "verified" in response.lower() or "fact check" in response.lower():
                reward += Fact_based_Check_reward
                
            # Check for knowledge base usage signals
            if "my knowledge" in response.lower() or "I know that" in response.lower():
                reward += knowledge_base_Check_reward
                
            # Check for search usage signals
            if "search results" in response.lower() or "found information" in response.lower():
                reward += Search_Check_reward
                
            # Check for confidence level signals
            if "confidence level" in response.lower() or "I am certain" in response.lower() or "I am uncertain" in response.lower():
                reward += Verify_check_reward
        
        flame_guard_reward = Verify_check_reward + Search_Check_reward + knowledge_base_Check_reward + Fact_based_Check_reward
        
        # 11. Latent Space Efficiency Reward
        latent_efficiency_reward = 0.0
        if loss is not None and binary_mask is not None:
            # Use the compute_coconut_reward function to calculate latent space efficiency
            raw_latent_reward = compute_coconut_reward(
                loss=loss,
                binary_mask=binary_mask,
                penalty_weight=self.latent_penalty_weight
            )
            
            # Scale the reward by the configured weight
            latent_efficiency_reward = raw_latent_reward.item() * self.latent_space_efficiency_weight
            
            # Log latent space usage
            if binary_mask is not None:
                effective_latent_count = binary_mask.sum(dim=1).squeeze(-1).mean().item()
                if effective_latent_count > 1:
                    penalty_per_extra_latent = self.latent_penalty_weight * self.latent_space_efficiency_weight
                    print(f"Using {effective_latent_count:.1f} latent spaces. Penalty for each space beyond the first: {penalty_per_extra_latent:.2f}")
                    print(f"Total latent space penalty: {penalty_per_extra_latent * (effective_latent_count - 1):.2f}")
                else:
                    print(f"Optimal latent space usage: Using only {effective_latent_count:.1f} latent space")
            
            reward += latent_efficiency_reward
        
        # --- Combine all rewards ---
        accuracy_reward = 0.0  # Use the value set in the accuracy reward section above if it exists
        if query is not None and response is not None and correct_response_checker is not None:
            if correct_response_checker(query, response):
                accuracy_reward = 10.0 * self.accuracy_reward_weight
            else:
                accuracy_reward = -2.0 * self.accuracy_reward_weight
        
        total_env_reward = reward + reasoning_quality_reward + accuracy_reward + repetition_penalty + brevity_reward + flame_guard_reward + tag_correct_reward + latent_efficiency_reward

        if reasoning_quality_reward > 0:
            print(f"R_env: Positive Reinforcement - Reasoning Quality Reward! Bonus: {reasoning_quality_reward:.2f}")
        elif reasoning_quality_reward < 0: # If reasoning quality is somehow penalized (less common, but possible)
                print(f"R_env: Negative Reinforcement - Reasoning Quality Penalty! Penalty: {reasoning_quality_reward:.2f}")
        
        # Log latent space efficiency reward
        if latent_efficiency_reward != 0.0:
            if latent_efficiency_reward > 0:
                print(f"R_env: Positive Reinforcement - Latent Space Efficiency Reward! Bonus: {latent_efficiency_reward:.2f}")
            else:
                print(f"R_env: Negative Reinforcement - Latent Space Efficiency Penalty! Penalty: {latent_efficiency_reward:.2f}")

        return total_env_reward


class DeepSeekDatasetProcessor:
    """
    Processor for preparing the pre-distillation dataset for GPRO training.
    This handles loading, preprocessing, and augmenting the thinking chains dataset.
    """
    def __init__(self, data_path: str = None):
        """
        Initialize the dataset processor.
        
        Args:
            data_path: Path to raw data
        """
        self.data_path = data_path
        self.processed_data = []
    
    def load_raw_data(self):
        """Load raw data from the specified path."""
        if not self.data_path or not os.path.exists(self.data_path):
            logger.warning("No data path provided or file not found")
            return []
            
        try:
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
                
            logger.info(f"Loaded {len(raw_data)} raw data examples")
            return raw_data
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return []
    
    def preprocess_thinking_chains(self, raw_data):
        """
        Preprocess thinking chains data from raw format to training format.
        
        Args:
            raw_data: Raw data loaded from file
            
        Returns:
            Preprocessed data ready for training
        """
        processed = []
        
        for item in raw_data:
            # Extract query and thinking chain
            query = item.get('question', '')
            thinking = item.get('thinking', '')
            answer = item.get('answer', '')
            
            # Format as needed for GPRO training
            processed_item = {
                'query': query,
                'thinking_chain': thinking,
                'answer': answer,
                # Add any additional metadata needed
                'metadata': {
                    'source': item.get('source', 'unknown'),
                    'category': item.get('category', 'general')
                }
            }
            
            processed.append(processed_item)
        
        logger.info(f"Preprocessed {len(processed)} thinking chain examples")
        self.processed_data = processed
        return processed
    
    def augment_data(self, augmentation_factor=1):
        """
        Apply simple augmentation techniques to increase dataset size.
        
        Args:
            augmentation_factor: Factor by which to augment the dataset
            
        Returns:
            Augmented dataset
        """
        if not self.processed_data:
            logger.warning("No processed data to augment")
            return []
        
        augmented = list(self.processed_data)  # Start with original data
        
        # Simple augmentation: add variations of existing examples
        for _ in range(augmentation_factor - 1):
            for item in self.processed_data:
                # Create a variation of the item
                # This is a placeholder - in practice, use more sophisticated augmentation
                augmented_item = {
                    'query': self._augment_text(item['query']),
                    'thinking_chain': item['thinking_chain'],  # Keep original thinking
                    'answer': item['answer'],  # Keep original answer
                    'metadata': {
                        **item['metadata'],
                        'augmented': True
                    }
                }
                
                augmented.append(augmented_item)
        
        logger.info(f"Augmented dataset to {len(augmented)} examples")
        return augmented
    
    def _augment_text(self, text):
        """
        Apply simple augmentation to text.
        
        Args:
            text: Original text
            
        Returns:
            Augmented text
        """
        # This is a simple placeholder implementation
        # In practice, use techniques like:
        # - Synonym replacement
        # - Random insertion/deletion
        # - Sentence restructuring
        
        # For this example, just add a prefix or change wording slightly
        import random
        
        prefixes = [
            "Could you tell me about ",
            "I'd like to know about ",
            "Please explain ",
            "Help me understand ",
            "I'm curious about "
        ]
        
        if random.random() < 0.5:
            # Add a prefix if the text doesn't already have one
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text
            
            selected_prefix = random.choice(prefixes)
            if text.endswith("?"):
                text = text[:-1]  # Remove question mark
                return selected_prefix + text.lower() + "?"
            else:
                return selected_prefix + text.lower()
        else:
            # Simple word replacement (very basic)
            replacements = {
                "how": "in what way",
                "why": "for what reason",
                "what": "which thing",
                "explain": "describe",
                "tell": "inform"
            }
            
            for original, replacement in replacements.items():
                if f" {original} " in text:
                    return text.replace(f" {original} ", f" {replacement} ")
            
            return text
    
    def save_processed_data(self, output_path):
        """
        Save processed data to file.
        
        Args:
            output_path: Path to save processed data
        """
        if not self.processed_data:
            logger.warning("No processed data to save")
            return
            
        try:
            with open(output_path, 'w') as f:
                json.dump(self.processed_data, f, indent=2)
                
            logger.info(f"Saved {len(self.processed_data)} processed examples to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def create_dataloader(self, batch_size=16, shuffle=True):
        """
        Create a DataLoader from processed data.
        
        Args:
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the processed data
        """
        # Convert processed data to dataset
        dataset = ThinkingChainsDataset(None)
        dataset.examples = self.processed_data
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: Batch of examples
            
        Returns:
            Collated batch
        """
        # Extract queries from batch
        queries = [item['query'] for item in batch]
        
        # Return batch in format expected by GPROSelfTaskTrainer
        return {
            'queries': queries,
            'items': batch  # Include full items for reference if needed
        }


def main():
    """
    Main function to demonstrate training the self-task component.
    
    This is a placeholder implementation. To use it in practice, you would need
    to instantiate your COCONUT model and appropriate reward calculator.
    """
    # This function is a placeholder for demonstration purposes
    logger.info("Starting self-task training with GPRO")
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train self-task component with GPRO")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="gpro_checkpoints", help="Directory to save outputs")
    parser.add_argument("--group_size", type=int, default=8, help="GPRO group size (G)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GPRO clipping parameter (ε)")
    parser.add_argument("--beta", type=float, default=0.01, help="KL regularization coefficient (β)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # In a real implementation, you would:
    # 1. Create/load your COCONUT model
    # model = YourCOCONUTModel(...)
    
    # 2. Initialize EnvironmentalRewardCalculator
    # reward_calculator = EnvironmentalRewardCalculator()
    
    # 3. Create GPROSelfTaskTrainer
    # trainer = GPROSelfTaskTrainer(
    #     model=model,
    #     reward_calculator=reward_calculator,
    #     group_size=args.group_size,
    #     epsilon=args.epsilon,
    #     beta=args.beta,
    #     learning_rate=args.learning_rate,
    #     checkpoint_dir=args.output_dir
    # )
    
    # 4. Load and preprocess training data
    # processor = DeepSeekDatasetProcessor(args.data_path)
    # raw_data = processor.load_raw_data()
    # processed_data = processor.preprocess_thinking_chains(raw_data)
    # dataloader = processor.create_dataloader(batch_size=args.batch_size)
    
    # 5. Train the model
    # trainer.train(
    #     dataloader=dataloader,
    #     num_epochs=args.num_epochs
    # )
    
    logger.info("Self-task training complete")
    
    # Play sound to indicate completion
    try:
        import platform
        import subprocess
        
        sound_file = "Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav"
        
        if platform.system() == "Linux":
            subprocess.run(["aplay", sound_file])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", sound_file])
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
        logger.info(f"Sound played: {sound_file}")
    except Exception as e:
        logger.error(f"Failed to play sound: {e}")


if __name__ == "__main__":
    main()


def compute_coconut_reward(loss: torch.Tensor, binary_mask: torch.Tensor, penalty_weight: float = 0.1) -> torch.Tensor:
    """
    Compute reward for the COCONUT model based on its accuracy loss and latent space usage.
    This function penalizes the model for each latent space used beyond the first one,
    encouraging the use of the minimum number of latent spaces required for an accurate answer.

    Args:
      loss: A scalar torch.Tensor representing the model's loss (accuracy measure, lower is better).
      binary_mask: A binary torch.Tensor of shape (batch_size, seq_len, 1) from the BinaryPatchingModule,
                   where each 1 indicates a patch boundary.
      penalty_weight: A float weight used to scale the penalty for each additional latent space.

    Returns:
      A torch.Tensor scalar reward value computed as:
         reward = -loss - penalty_weight * mean(max(0, effective_latent_count - 1))
         
    This means for each latent space beyond the first one, the model receives a penalty of
    penalty_weight. The more latent spaces used, the greater the penalty.
    """
    effective_latent_count = binary_mask.sum(dim=1).squeeze(-1)  # shape: (batch_size,)
    # Penalize each latent space beyond the first one
    excess_latents = torch.clamp(effective_latent_count - 1, min=0)
    penalty = penalty_weight * excess_latents.mean()
    reward = -loss - penalty
    return reward