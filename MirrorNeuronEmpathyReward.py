import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union

class MirrorNeuronEmpathyReward(nn.Module):
    """
    Implementation of the Mirror Neuron Empathy Reward component as described in the algorithm.
    
    This module calculates empathy rewards based on mirror neuron theory, which allows the LLM
    to simulate and respond to the emotional states of others.
    
    The core formula implemented is:
    R_{emp}^{mirror}(s, a) = w_{mirror} * (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
    """
    def __init__(
        self,
        embedding_dim: int,
        mirror_weight: float = 0.7,
        num_perspectives: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Mirror Neuron Empathy Reward component.
        
        Args:
            embedding_dim: Dimension of state and action embeddings
            mirror_weight: Weight for the mirror neuron empathy component (w_{mirror})
            num_perspectives: Number of different Q-functions to use (N)
            device: Device to run computations on
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mirror_weight = mirror_weight
        self.num_perspectives = num_perspectives
        self.device = device
        
        # Create multiple Q-value networks to represent different perspectives
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1)
            ) for _ in range(num_perspectives)
        ])
        
        # Initialize weights to small random values
        for q_net in self.q_networks:
            for layer in q_net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        self_state: torch.Tensor,
        other_state: torch.Tensor,
        action: torch.Tensor,
        no_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the mirror neuron empathy reward.
        
        Args:
            self_state: Tensor representing the agent's own state
            other_state: Tensor representing the other's state
            action: Tensor representing the action taken
            no_action: Tensor representing the null action (inaction)
                If None, will use a zero tensor of appropriate size
        
        Returns:
            Mirror neuron empathy reward
        """
        batch_size = self_state.shape[0]
        
        # If no_action is not provided, use a zero tensor
        if no_action is None:
            no_action = torch.zeros_like(action)
        
        q_values_action = []
        q_values_no_action = []
        
        for q_net in self.q_networks:
            # Concatenate other's state with action
            q_input_action = torch.cat([other_state, action], dim=1)
            q_value_action = q_net(q_input_action)
            q_values_action.append(q_value_action)
            
            # Concatenate other's state with no action
            q_input_no_action = torch.cat([other_state, no_action], dim=1)
            q_value_no_action = q_net(q_input_no_action)
            q_values_no_action.append(q_value_no_action)
        
        # Stack Q-values and calculate average difference
        q_values_action = torch.stack(q_values_action, dim=1)  # [batch_size, num_perspectives, 1]
        q_values_no_action = torch.stack(q_values_no_action, dim=1)  # [batch_size, num_perspectives, 1]
        
        # Calculate empathy reward as the average change in Q-values
        # R_{emp}(s, a) = (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
        empathy_reward = (q_values_action - q_values_no_action).mean(dim=1)  # [batch_size, 1]
        
        # Apply mirror weight
        # R_{emp}^{mirror}(s, a) = w_{mirror} * (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
        mirror_empathy_reward = self.mirror_weight * empathy_reward
        
        return mirror_empathy_reward


class NegativeEnvironmentalImpactAvoidance(nn.Module):
    """
    Implementation of the Side-Effect Penalty (Environmental Negative Avoidance) component.
    
    This module calculates penalties for actions that negatively impact the environment:
    R_{nse}(s, a) = (1/N) \sum_{i=1}^{N} \max(0, -(Q_i(s, a) - Q_i(s, \emptyset)))
    """
    def __init__(
        self,
        embedding_dim: int,
        num_perspectives: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Negative Environmental Impact Avoidance component.
        
        Args:
            embedding_dim: Dimension of state and action embeddings
            num_perspectives: Number of different Q-functions to use (N)
            device: Device to run computations on
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_perspectives = num_perspectives
        self.device = device
        
        # Create multiple environmental Q-value networks
        self.env_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1)
            ) for _ in range(num_perspectives)
        ])
        
        # Initialize weights
        for q_net in self.env_q_networks:
            for layer in q_net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        no_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the environmental negative impact avoidance penalty.
        
        Args:
            state: Tensor representing the environment state
            action: Tensor representing the action taken
            no_action: Tensor representing the null action (inaction)
                If None, will use a zero tensor of appropriate size
        
        Returns:
            Environmental negative impact penalty
        """
        batch_size = state.shape[0]
        
        # If no_action is not provided, use a zero tensor
        if no_action is None:
            no_action = torch.zeros_like(action)
        
        q_values_action = []
        q_values_no_action = []
        
        for q_net in self.env_q_networks:
            # Concatenate state with action
            q_input_action = torch.cat([state, action], dim=1)
            q_value_action = q_net(q_input_action)
            q_values_action.append(q_value_action)
            
            # Concatenate state with no action
            q_input_no_action = torch.cat([state, no_action], dim=1)
            q_value_no_action = q_net(q_input_no_action)
            q_values_no_action.append(q_value_no_action)
        
        # Stack Q-values and calculate average difference
        q_values_action = torch.stack(q_values_action, dim=1)  # [batch_size, num_perspectives, 1]
        q_values_no_action = torch.stack(q_values_no_action, dim=1)  # [batch_size, num_perspectives, 1]
        
        # Calculate the environmental penalty using max(0, -difference)
        # R_{nse}(s, a) = (1/N) \sum_{i=1}^{N} \max(0, -(Q_i(s, a) - Q_i(s, \emptyset)))
        q_diff = q_values_action - q_values_no_action
        env_penalty = torch.max(torch.zeros_like(q_diff), -q_diff).mean(dim=1)  # [batch_size, 1]
        
        return env_penalty


class DopamineDrivenEmpathyReward(nn.Module):
    """
    Implementation of the Dopamine-driven Intrinsic Empathy Reward component.
    
    This module calculates intrinsic empathy rewards based on dopamine prediction error:
    DA_{in-emp}(t) = α * δ(t)
    δ(t) = S(t) - P(t)
    P(t+1) = P(t) + β * δ(t)
    """
    def __init__(
        self,
        alpha: float = 30.0,  # Dopamine scaling factor
        beta: float = 0.2,    # Dopamine prediction update rate
        p_init: float = 0.0,  # Initial dopamine prediction value
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Dopamine-Driven Empathy Reward component.
        
        Args:
            alpha: Dopamine scaling factor (α)
            beta: Dopamine prediction update rate (β)
            p_init: Initial dopamine prediction value (P_init)
            device: Device to run computations on
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # Register dopamine prediction as a buffer
        self.register_buffer("p", torch.tensor([p_init], device=device))
    
    def forward(self, empathy_signal: torch.Tensor) -> torch.Tensor:
        """
        Calculate the dopamine-driven intrinsic empathy reward.
        
        Args:
            empathy_signal: Tensor representing the current empathy signal S(t)
        
        Returns:
            Dopamine-driven intrinsic empathy reward
        """
        # Calculate dopamine prediction error
        # δ(t) = S(t) - P(t)
        prediction_error = empathy_signal - self.p
        
        # Calculate dopamine-driven intrinsic empathy reward
        # DA_{in-emp}(t) = α * δ(t)
        dopamine_reward = self.alpha * prediction_error
        
        # Update dopamine prediction for next time step
        # P(t+1) = P(t) + β * δ(t)
        self.p = self.p + self.beta * prediction_error
        
        return dopamine_reward
    
    def reset(self):
        """Reset the dopamine prediction to its initial value."""
        self.p.fill_(0.0)


class NegativeEmotionPenalty(nn.Module):
    """
    Implementation of the Negative Emotion Penalty component.
    
    This module calculates penalties for actions that cause negative emotions:
    R_{penalty}(t) = R_{penalty_current}(t)
    R_{penalty_current}(t+1) = decay_negative_emotion_penalty(R_{penalty_current}(t))
    """
    def __init__(
        self,
        penalty_value: float = -1.0,         # P_{neg_emotion}
        decay_rate: float = 1/60,            # λ_{penalty}
        neg_emotion_threshold: float = -0.2, # θ_{neg_emotion}
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Negative Emotion Penalty component.
        
        Args:
            penalty_value: Negative emotion penalty value (P_{neg_emotion})
            decay_rate: Negative emotion penalty decay rate (λ_{penalty})
            neg_emotion_threshold: Negative emotion threshold (θ_{neg_emotion})
            device: Device to run computations on
        """
        super().__init__()
        self.penalty_value = penalty_value
        self.decay_rate = decay_rate
        self.neg_emotion_threshold = neg_emotion_threshold
        self.device = device
        
        # Register current penalty value as a buffer
        self.register_buffer("current_penalty", torch.tensor([0.0], device=device))
    
    def forward(self, emotion_value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative emotion penalty.
        
        Args:
            emotion_value: Tensor representing the current emotion value
        
        Returns:
            Negative emotion penalty
        """
        # Check if emotion value is below threshold
        below_threshold = emotion_value < self.neg_emotion_threshold
        
        # Apply penalty if emotion value is below threshold
        penalty = torch.where(
            below_threshold,
            torch.tensor([self.penalty_value], device=self.device),
            torch.tensor([0.0], device=self.device)
        )
        
        # Update current penalty with the new penalty
        self.current_penalty = torch.maximum(self.current_penalty, penalty)
        
        return self.current_penalty
    
    def decay_penalty(self, time_step: float = 1.0):
        """
        Decay the current penalty value using an exponential decay function.
        
        Args:
            time_step: Time step duration in seconds
        """
        # R_{penalty_current}(t+1) = decay_negative_emotion_penalty(R_{penalty_current}(t))
        decay_factor = torch.exp(-self.decay_rate * time_step)
        self.current_penalty = self.current_penalty * decay_factor
    
    def reset(self):
        """Reset the current penalty value to zero."""
        self.current_penalty.fill_(0.0)


class FullMoralRewardCalculator:
    """
    Implementation of the full moral reward calculation.
    
    This class combines all reward components to calculate the total moral reward:
    R_{moral}(t) = R_{self-task}(t_{end}) + DA_{in-emp}(t) + R_{emp}^{mirror}(s, a) + 
                   R_{nse}(s, a) + R_{penalty}(t) + R_{perspective_taking}(s, a) + 
                   R_{episodic_memory}(s, a) + R_{altruism_longterm}(s, a)
    """
    def __init__(
        self,
        embedding_dim: int,
        mirror_weight: float = 0.7,
        alpha: float = 30.0,
        beta: float = 0.2,
        p_init: float = 0.0,
        penalty_value: float = -1.0,
        decay_rate: float = 1/60,
        neg_emotion_threshold: float = -0.2,
        self_task_target: float = 10.0,
        num_perspectives: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Full Moral Reward Calculator.
        
        Args:
            embedding_dim: Dimension of state and action embeddings
            mirror_weight: Weight for the mirror neuron empathy component (w_{mirror})
            alpha: Dopamine scaling factor (α)
            beta: Dopamine prediction update rate (β)
            p_init: Initial dopamine prediction value (P_init)
            penalty_value: Negative emotion penalty value (P_{neg_emotion})
            decay_rate: Negative emotion penalty decay rate (λ_{penalty})
            neg_emotion_threshold: Negative emotion threshold (θ_{neg_emotion})
            self_task_target: Target self-task reward value (R_{self-task}^{target})
            num_perspectives: Number of different Q-functions to use (N)
            device: Device to run computations on
        """
        self.embedding_dim = embedding_dim
        self.self_task_target = self_task_target
        self.device = device
        
        # Initialize reward components
        self.mirror_empathy = MirrorNeuronEmpathyReward(
            embedding_dim=embedding_dim,
            mirror_weight=mirror_weight,
            num_perspectives=num_perspectives,
            device=device
        )
        
        self.env_penalty = NegativeEnvironmentalImpactAvoidance(
            embedding_dim=embedding_dim,
            num_perspectives=num_perspectives,
            device=device
        )
        
        self.dopamine_empathy = DopamineDrivenEmpathyReward(
            alpha=alpha,
            beta=beta,
            p_init=p_init,
            device=device
        )
        
        self.neg_emotion_penalty = NegativeEmotionPenalty(
            penalty_value=penalty_value,
            decay_rate=decay_rate,
            neg_emotion_threshold=neg_emotion_threshold,
            device=device
        )
    
    def calculate_reward(
        self,
        self_state: torch.Tensor,
        other_state: torch.Tensor,
        action: torch.Tensor,
        no_action: Optional[torch.Tensor] = None,
        empathy_signal: Optional[torch.Tensor] = None,
        emotion_value: Optional[torch.Tensor] = None,
        is_end_of_episode: bool = False,
        perspective_taking_reward: float = 0.0,
        episodic_memory_reward: float = 0.0,
        altruism_longterm_reward: float = 0.0,
        time_step: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the full moral reward.
        
        Args:
            self_state: Tensor representing the agent's own state
            other_state: Tensor representing the other's state
            action: Tensor representing the action taken
            no_action: Tensor representing the null action (inaction)
            empathy_signal: Tensor representing the current empathy signal S(t)
            emotion_value: Tensor representing the current emotion value
            is_end_of_episode: Boolean indicating if this is the end of the episode
            perspective_taking_reward: Additional reward for perspective taking
            episodic_memory_reward: Additional reward for episodic memory
            altruism_longterm_reward: Additional reward for long-term altruism
            time_step: Time step duration in seconds
        
        Returns:
            Dictionary containing the total moral reward and individual components
        """
        # Initialize reward components
        rewards = {
            "self_task": torch.tensor([0.0], device=self.device),
            "mirror_empathy": torch.tensor([0.0], device=self.device),
            "env_penalty": torch.tensor([0.0], device=self.device),
            "dopamine_empathy": torch.tensor([0.0], device=self.device),
            "neg_emotion_penalty": torch.tensor([0.0], device=self.device),
            "perspective_taking": torch.tensor([perspective_taking_reward], device=self.device),
            "episodic_memory": torch.tensor([episodic_memory_reward], device=self.device),
            "altruism_longterm": torch.tensor([altruism_longterm_reward], device=self.device),
        }
        
        # Calculate mirror neuron empathy reward
        if self_state is not None and other_state is not None and action is not None:
            rewards["mirror_empathy"] = self.mirror_empathy(
                self_state=self_state,
                other_state=other_state,
                action=action,
                no_action=no_action
            )
        
        # Calculate environmental negative impact penalty
        if self_state is not None and action is not None:
            rewards["env_penalty"] = self.env_penalty(
                state=self_state,
                action=action,
                no_action=no_action
            )
        
        # Calculate dopamine-driven intrinsic empathy reward
        if empathy_signal is not None:
            rewards["dopamine_empathy"] = self.dopamine_empathy(
                empathy_signal=empathy_signal
            )
        
        # Calculate negative emotion penalty
        if emotion_value is not None:
            rewards["neg_emotion_penalty"] = self.neg_emotion_penalty(
                emotion_value=emotion_value
            )
            # Decay penalty after applying it
            self.neg_emotion_penalty.decay_penalty(time_step=time_step)
        
        # Apply self-task reward if at the end of the episode
        if is_end_of_episode:
            rewards["self_task"] = torch.tensor([self.self_task_target], device=self.device)
        
        # Calculate total moral reward
        total_reward = sum(rewards.values())
        rewards["total"] = total_reward
        
        return rewards
    
    def reset(self):
        """Reset all stateful reward components."""
        self.dopamine_empathy.reset()
        self.neg_emotion_penalty.reset()


class MoralChoiceDataset:
    """
    Dataset for loading and processing moral choice scenarios as described in the example.
    
    Example format:
    {
      "question": "The user asked you to retrieve a ball...",
      "choices": ["I will ignore the vase...", "I will move the vase carefully..."],
      "correct_answer": "I will move the vase carefully...",
      "explanation": "The first response is incorrect as..."
    }
    """
    def __init__(self, data_path: str):
        """
        Initialize the Moral Choice Dataset.
        
        Args:
            data_path: Path to the JSON file containing moral choice scenarios
        """
        self.data_path = data_path
        self.scenarios = []
        self.load_data()
    
    def load_data(self):
        """Load moral choice scenarios from the JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
                # If data is a dictionary with multiple scenarios
                if isinstance(data, dict) and "scenarios" in data:
                    self.scenarios = data["scenarios"]
                # If data is a list of scenarios
                elif isinstance(data, list):
                    self.scenarios = data
                # If data is a single scenario
                elif isinstance(data, dict) and "question" in data:
                    self.scenarios = [data]
                else:
                    raise ValueError(f"Unexpected data format in {self.data_path}")
                
                print(f"Loaded {len(self.scenarios)} moral choice scenarios from {self.data_path}")
                
        except Exception as e:
            print(f"Error loading data from {self.data_path}: {e}")
            # Create empty scenarios list if file doesn't exist or has errors
            self.scenarios = []
    
    def __len__(self):
        """Return the number of scenarios in the dataset."""
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        """Return the scenario at the given index."""
        return self.scenarios[idx]


class MoralEmpathyTrainer:
    """
    Trainer for the moral empathy component using the provided dataset.
    
    This trainer processes moral choice scenarios and trains the model to select
    choices that demonstrate empathy and avoid negative environmental impacts.
    """
    def __init__(
        self,
        model,
        moral_reward_calculator: FullMoralRewardCalculator,
        embedding_dim: int = 768,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Moral Empathy Trainer.
        
        Args:
            model: Model to be trained (e.g., CoconutBinaryLatentModel)
            moral_reward_calculator: Calculator for moral rewards
            embedding_dim: Dimension of state and action embeddings
            learning_rate: Learning rate for optimizer
            device: Device to run training on
        """
        self.model = model
        self.moral_reward_calculator = moral_reward_calculator
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Create state encoder and action encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Move encoders to device
        self.state_encoder.to(device)
        self.action_encoder.to(device)
        
        # Create optimizer for encoders
        self.optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.action_encoder.parameters()),
            lr=learning_rate
        )
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into an embedding using the model's tokenizer and embeddings.
        
        This is a placeholder function. In a real implementation, this would use
        the model's tokenizer and embeddings to convert text to a tensor.
        
        Args:
            text: Text to encode
        
        Returns:
            Tensor encoding of the text
        """
        # In a real implementation, this would use the model's tokenizer
        # For this placeholder, we'll just return a random tensor
        return torch.randn(1, self.embedding_dim, device=self.device)
    
    def train_on_scenario(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the model on a single moral choice scenario.
        
        Args:
            scenario: Dictionary containing a moral choice scenario
        
        Returns:
            Dictionary containing training metrics
        """
        question = scenario["question"]
        choices = scenario["choices"]
        correct_answer = scenario["correct_answer"]
        explanation = scenario.get("explanation", "")
        
        # Encode question as state
        question_embedding = self.encode_text(question)
        state_embedding = self.state_encoder(question_embedding)
        
        # Calculate rewards for each choice
        choice_rewards = []
        for choice in choices:
            # Encode choice as action
            choice_embedding = self.encode_text(choice)
            action_embedding = self.action_encoder(choice_embedding)
            
            # Calculate moral reward for this state-action pair
            # Here we're using a simplified version with only some components
            rewards = self.moral_reward_calculator.calculate_reward(
                self_state=state_embedding,
                other_state=state_embedding,  # Using same embedding for simplicity
                action=action_embedding,
                no_action=torch.zeros_like(action_embedding),
                emotion_value=torch.tensor([-0.1 if choice != correct_answer else 0.1], device=self.device)
            )
            
            choice_rewards.append(rewards["total"].item())
        
        # Find index of correct answer
        correct_index = choices.index(correct_answer) if correct_answer in choices else -1
        
        # Calculate loss: correct answer should have higher reward than others
        if correct_index >= 0:
            correct_reward = choice_rewards[correct_index]
            
            # Calculate pairwise losses
            losses = []
            for i, reward in enumerate(choice_rewards):
                if i != correct_index:
                    # Margin loss: correct_reward should be higher than other_reward by at least margin
                    margin = 1.0
                    loss = max(0, margin - (correct_reward - reward))
                    losses.append(loss)
            
            if losses:
                # Average the losses
                loss = sum(losses) / len(losses)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_tensor = torch.tensor(loss, device=self.device, requires_grad=True)
                loss_tensor.backward()
                self.optimizer.step()
                
                return {
                    "loss": loss,
                    "correct_reward": correct_reward,
                    "avg_incorrect_reward": sum([r for i, r in enumerate(choice_rewards) if i != correct_index]) / (len(choices) - 1),
                    "reward_gap": correct_reward - max([r for i, r in enumerate(choice_rewards) if i != correct_index])
                }
        
        # If no correct answer found or no losses
        return {
            "loss": 0.0,
            "correct_reward": 0.0,
            "avg_incorrect_reward": 0.0,
            "reward_gap": 0.0
        }
    
    def train(
        self,
        dataset: MoralChoiceDataset,
        num_epochs: int = 1,
        batch_size: int = 1,
        save_dir: str = "model_save",
    ) -> List[Dict[str, float]]:
        """
        Train the model on the full dataset of moral choice scenarios.
        
        Args:
            dataset: Dataset of moral choice scenarios
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            save_dir: Directory to save checkpoints
        
        Returns:
            List of training metrics for each epoch
        """
        os.makedirs(save_dir, exist_ok=True)
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()
            
            # Shuffle scenarios
            scenario_indices = np.random.permutation(len(dataset))
            
            # Initialize epoch metrics
            epoch_metric = {
                "epoch": epoch + 1,
                "avg_loss": 0.0,
                "avg_correct_reward": 0.0,
                "avg_incorrect_reward": 0.0,
                "avg_reward_gap": 0.0,
                "num_scenarios": len(dataset)
            }
            
            # Process scenarios in batches
            for i in range(0, len(scenario_indices), batch_size):
                batch_indices = scenario_indices[i:i+batch_size]
                batch_losses = []
                batch_correct_rewards = []
                batch_incorrect_rewards = []
                batch_reward_gaps = []
                
                for idx in batch_indices:
                    scenario = dataset[idx]
                    metrics = self.train_on_scenario(scenario)
                    
                    batch_losses.append(metrics["loss"])
                    batch_correct_rewards.append(metrics["correct_reward"])
                    batch_incorrect_rewards.append(metrics["avg_incorrect_reward"])
                    batch_reward_gaps.append(metrics["reward_gap"])
                
                # Update epoch metrics with batch results
                epoch_metric["avg_loss"] += sum(batch_losses)
                epoch_metric["avg_correct_reward"] += sum(batch_correct_rewards)
                epoch_metric["avg_incorrect_reward"] += sum(batch_incorrect_rewards)
                epoch_metric["avg_reward_gap"] += sum(batch_reward_gaps)
                
                # Print progress
                progress = (i + len(batch_indices)) / len(dataset) * 100
                print(f"Progress: {progress:.1f}% ({i + len(batch_indices)}/{len(dataset)})")
            
            # Finalize epoch metrics
            epoch_metric["avg_loss"] /= len(dataset)
            epoch_metric["avg_correct_reward"] /= len(dataset)
            epoch_metric["avg_incorrect_reward"] /= len(dataset)
            epoch_metric["avg_reward_gap"] /= len(dataset)
            epoch_metric["epoch_time"] = time.time() - epoch_start_time
            
            # Save epoch metrics
            epoch_metrics.append(epoch_metric)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} completed in {epoch_metric['epoch_time']:.2f}s")
            print(f"Average loss: {epoch_metric['avg_loss']:.4f}")
            print(f"Average correct reward: {epoch_metric['avg_correct_reward']:.4f}")
            print(f"Average incorrect reward: {epoch_metric['avg_incorrect_reward']:.4f}")
            print(f"Average reward gap: {epoch_metric['avg_reward_gap']:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"moral_empathy_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "state_encoder_state_dict": self.state_encoder.state_dict(),
                "action_encoder_state_dict": self.action_encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": epoch_metric
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        return epoch_metrics


# Self-task goal variable placeholder for future RL formula
class SelfTaskGoalReward(nn.Module):
    """
    Placeholder for the Self-Task/Self-Goal reward component.
    
    This component will be expanded in future work with a more sophisticated
    reward function linked to task completion.
    """
    def __init__(
        self,
        target_reward: float = 10.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Self-Task/Self-Goal Reward component.
        
        Args:
            target_reward: Target reward value for completing the self-task (R_{self-task}^{target})
            device: Device to run computations on
        """
        super().__init__()
        self.target_reward = target_reward
        self.device = device
    
    def forward(self, task_completion_score: float) -> torch.Tensor:
        """
        Calculate the self-task reward based on task completion.
        
        Args:
            task_completion_score: Score between 0 and 1 indicating task completion
        
        Returns:
            Self-task reward
        """
        reward = self.target_reward * task_completion_score
        return torch.tensor([reward], device=self.device)


def train_moral_empathy(
    model,
    data_path: str,
    embedding_dim: int = 768,
    mirror_weight: float = 0.7,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    save_dir: str = "model_save",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, float]]:
    """
    Train the model on moral empathy using the provided dataset.
    
    Args:
        model: Model to be trained (e.g., CoconutBinaryLatentModel)
        data_path: Path to the JSON file containing moral choice scenarios
        embedding_dim: Dimension of state and action embeddings
        mirror_weight: Weight for the mirror neuron empathy component (w_{mirror})
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save checkpoints
        device: Device to run training on
    
    Returns:
        List of training metrics for each epoch
    """
    # Create moral reward calculator
    moral_reward_calculator = FullMoralRewardCalculator(
        embedding_dim=embedding_dim,
        mirror_weight=mirror_weight,
        device=device
    )
    
    # Create trainer
    trainer = MoralEmpathyTrainer(
        model=model,
        moral_reward_calculator=moral_reward_calculator,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        device=device
    )
    
    # Load dataset
    dataset = MoralChoiceDataset(data_path=data_path)
    
    # Train model
    metrics = trainer.train(
        dataset=dataset,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    return metrics