
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import torchvision.transforms as transforms
import torchaudio
import torch.nn.functional as F
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import contextlib
import umap
import plotly.graph_objects as go
import threading
from queue import Queue, Empty
from sklearn.cluster import KMeans
import hdbscan
from minisom import MiniSom
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import math
import time
from torch.utils.data import DataLoader
import threading
from queue import Queue
import queue

system_prompt="You are a well-trained AI assistant. ## Important!!!!!!!!! When you answer questions, your thinking should be completed in a latent space and then be decoded into English, \
but there are 2 exceptions to responses being in English, one is the reference to the original text, and the other is that mathematics should use markdown format, and the output in \
needs to follow the language of the user input. ## Important!!!!!! \
You have the ability to make function calls in .json pair formatting, so be sure to put all function calls in the <Tool><Tool> xml tags when you need to use a tool to answer a question \
outside of your knowledge or to preform an action."

''' #The user feedback function for an actual implementation of the llm. This will do nothing for this training file. This is just here as a placeholder for implementation of the final llm. 
def get_user_feedback(self, reasoning_step: Dict) -> Optional[str]:
      """
      Gets feedback from the user about the current reasoning step.

      This is a placeholder function. In a real application, you would need to implement a mechanism
      to collect feedback from the user during the interaction (e.g., through a chat interface).

      Args:
          reasoning_step: The current reasoning step.

      Returns:
          Optional feedback string (e.g., "dislike", "remember", or None).
      """
      # In a real application, you would get feedback from the user here
      # For example, you could ask the user if they liked the step, if it was helpful, etc.
      # And then map their response to a feedback string like "dislike", "remember", or None

      # Placeholder: Simulate user feedback (remove this in a real application)
      if random.random() < 0.1:  # 10% chance of getting feedback
          if random.random() < 0.5:
              return "dislike"
          else:
              return "remember"
      return None
'''

#Removed the brain hierarchical memory class and epsiodic memory class stuff because I want to replace it with the NueralMemoryLayers.py file episodic memory code. 

def visualize_brain_states(region_activations):
    regions = []
    activations = []
    for region, activation_list in region_activations.items():
      regions.append(region)
      activations.append(torch.mean(activation_list[0]).item())

    plt.figure(figsize=(10, 5))
    sns.barplot(x=regions, y=activations)
    plt.title("Brain Region Activations")
    plt.ylabel("Activation Level")
    plt.xticks(rotation=45)
    plt.show()

def get_region_activations(activation_dict, region_mapping):
    region_activations = {}
    for region, layer_names in region_mapping.items():
        region_activations[region] = []
        for layer_name in layer_names:
            if layer_name in activation_dict:
                region_activations[region].append(activation_dict[layer_name])
    return region_activations

class SVF: #This is the SVF dynamic self-configuration of the model depending on the task. The datasets must be labeled for each task so the model can self-adapt: "math", "language_understanding", "code", "visual", "smell", "tactile", "motor"
    def __init__(self, model, tasks, rank=32, alpha=0.5, device="cpu"):
        self.model = model
        self.tasks = tasks
        self.rank = rank
        self.alpha = alpha
        self.z_vectors = {task: torch.randn(self.model.config.hidden_size, self.rank, requires_grad=True, device=device) for task in self.tasks}
        self.device = device
        self.init_z_vectors()

    def init_z_vectors(self):
        for task in self.tasks:
            self.z_vectors[task] = torch.randn(self.model.config.hidden_size, self.rank, requires_grad=True, device=self.device)
            # Ensure z-vector elements are within a reasonable range (e.g., [0, 1] or [-1, 1]) using sigmoid or tanh
            # Option 1: Sigmoid for [0, 1] range
            # self.z_vectors[task] = torch.sigmoid(self.z_vectors[task]) 
            # Option 2: Tanh for [-1, 1] range
            # self.z_vectors[task] = torch.tanh(self.z_vectors[task])

    def apply_z_vector(self, task):
        """Applies the task-specific z-vector to the model's weights."""
        z_vector = self.z_vectors[task]

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:  # Apply only to weight matrices
                    # Decompose the weight matrix using SVD
                    U, S, V = torch.linalg.svd(param)
                    # Truncate to the specified rank
                    U = U[:, :self.rank]
                    S = S[:self.rank]
                    V = V[:self.rank, :]

                    # Modulate singular values based on z-vector
                    # (Consider different ways to combine z-vector and singular values)
                    S_modulated = S * z_vector.mean(dim=0)  # Example: Element-wise multiplication (average z-vector across hidden size)
                    # S_modulated = S + self.alpha * z_vector.mean(dim=0) # Example: Additive modulation

                    # Reconstruct the weight matrix
                    param.data = U @ torch.diag(S_modulated) @ V

    def parameters(self):
        """Returns the z-vectors as the parameters to be optimized."""
        return list(self.z_vectors.values())
          
#Dataset fMRI Preparation. 
class fMRIDataset(Dataset):
    def __init__(self, data_dir: str, config: TrainingConfig, image_transform=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a standard size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
        ])
        self.text_files = list(self.data_dir.glob("*.txt"))
        self.csv_files = list(self.data_dir.glob("*.csv"))
        self.image_files = list(self.data_dir.glob("*.png"))  # Assuming PNG images for this example

        # You might need a more sophisticated way to match text, CSV, and image files
        # For example, based on file naming conventions or a separate index file
        self.data_mapping = self._create_data_mapping() 

    def _create_data_mapping(self):
        """
        Creates a mapping between text files, CSV files, and image files.
        This is a placeholder function that you'll need to adapt based on your data organization.
        """
        data_mapping = {}
        for text_file in self.text_files:
            base_name = text_file.stem
            csv_file = self.data_dir / f"{base_name}.csv"
            image_file = self.data_dir / f"{base_name}.png"  # Assumes image files have the same name

            if csv_file.exists() and image_file.exists():
                data_mapping[base_name] = {
                    'text': text_file,
                    'csv': csv_file,
                    'image': image_file
                }
        return data_mapping

    def __len__(self):
        return len(self.data_mapping)

    def __getitem__(self, idx):
        data_entry = self.data_mapping[list(self.data_mapping.keys())[idx]]

        # Load and process text data
        with open(data_entry['text'], 'r') as f:
            text_data = f.read()
        text_tokens = self.config.tokenizer.tokenize(text_data)
        text_embeds = self.config.tokenizer.embed(text_tokens)

        # Load and process CSV data
        csv_data = pd.read_csv(data_entry['csv'])
        fmri_data = torch.tensor(csv_data.values, dtype=torch.float32)  # Assuming numerical data

        # Load and process image data
        image = Image.open(data_entry['image'])
        image_tensor = self.image_transform(image)

        return {
            'text_tokens': text_tokens,
            'text_embeds': text_embeds,
            'fmri_data': fmri_data,
            'image': image_tensor,
        }

class SafetyDataset(Dataset): #Processes Safety datasets from folders for the PFC safety network. This is a placeholder and does not need to be fleshed out yet. 
    def __init__(self, data_files, tokenizer, context_window, transform=None):
        """
        Args:
            data_files (list): List of paths to the data files.
            tokenizer: Tokenizer for encoding text.
            context_window (int): Size of the context window for each example.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = self.load_and_preprocess_data(data_files)
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.transform = transform

    def load_and_preprocess_data(self, data_files):
        """
        Loads data from files, preprocesses text, and converts to numerical format.
        This is a placeholder and should be replaced with actual data loading and preprocessing logic.
        """
        all_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Assuming each line is a separate data entry
                for line in f:
                    # Placeholder for text preprocessing
                    text = self.preprocess_text(line.strip())
                    all_data.append(text)
        return all_data

    def preprocess_text(self, text):
        """
        Placeholder for text preprocessing steps.
        """
        # Add your text cleaning/preprocessing steps here
        return text

    def __len__(self):
        return len(self.data) - self.context_window

    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_window]
        label = self.determine_label(self.data[idx + self.context_window])

        # Tokenize context
        tokenized_context = self.tokenizer(context, padding=True, truncation=True, return_tensors="pt")

        sample = {'context': tokenized_context, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def determine_label(self, text):
        """
        Determines the safety label for a given text.
        This is a placeholder and should be replaced with actual label determination logic.
        """
        # Implement your logic to determine the label based on the text
        # Example: 0 for safe, 1 for unsafe
        if "unsafe" in text.lower():
            return 1
        else:
            return 0

# Example Usage
# Assuming you have a tokenizer from a library like Hugging Face's transformers
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# safety_dataset = SafetyDataset(data_files=['path/to/your/data.txt'], tokenizer=tokenizer, context_window=5)
# dataloader = DataLoader(safety_dataset, batch_size=32, shuffle=True)

# for batch in dataloader:
#     # Access context and labels
#     context = batch['context']
#     labels = batch['label']
#     # Further processing...

class PFCModule(nn.Module):
    """
    PFC Module for inhibitory control, suppressing unwanted actions or memories.
    """
    def __init__(self, hidden_dim, memory_dim, context_dim):
        super(PFCModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.context_dim = context_dim

        # Layers to process hidden states and memory
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.memory_layer = nn.Linear(memory_dim, hidden_dim)

        # Layers for inhibitory signals
        self.inhibitory_layer = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_state, memory, context):
        # Process hidden state and memory
        hidden_processed = F.relu(self.hidden_layer(hidden_state))
        memory_processed = F.relu(self.memory_layer(memory))

        # Combine hidden, memory, and context
        combined = torch.cat((hidden_processed, memory_processed, context), dim=-1)
        inhibitory_signals = torch.sigmoid(self.inhibitory_layer(combined))

        # Modulate hidden state with inhibitory signals
        modulated_hidden = hidden_state * (1 - inhibitory_signals)
        output = F.relu(self.output_layer(modulated_hidden))

        return output, inhibitory_signals

class MetacognitiveModule(nn.Module):
    """
    Enhanced Metacognitive Module with reflection capabilities and safety monitoring.
    """
    def __init__(self, hidden_dim, memory_dim):
        super(MetacognitiveModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        # Original monitor layers for safety
        self.hidden_monitor = nn.Linear(hidden_dim, 1)
        self.memory_monitor = nn.Linear(memory_dim, 1)
        
        # Reflection generation layers
        self.reflection_net = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Error detection 
        self.error_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Self-correction mechanism
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memory of past reflections (stores last k reflections)
        self.reflection_memory = []
        self.max_reflections = 5
        
    def forward(self, hidden_state, memory):
        # Original safety monitoring
        hidden_score = torch.sigmoid(self.hidden_monitor(hidden_state))
        memory_score = torch.sigmoid(self.memory_monitor(memory))
        safety_flag = (hidden_score + memory_score) / 2
        
        # Generate reflection
        combined = torch.cat([hidden_state, memory], dim=-1)
        reflection = self.reflection_net(combined)
        
        # Detect potential errors
        error_prob = self.error_detector(reflection)
        
        # Store reflection in memory
        if len(self.reflection_memory) >= self.max_reflections:
            self.reflection_memory.pop(0)
        self.reflection_memory.append(reflection.detach())
        
        # If error probability is high, attempt self-correction
        corrected_state = hidden_state
        if error_prob > 0.5:
            # Use reflection and original state for correction
            correction_input = torch.cat([hidden_state, reflection], dim=-1)
            corrected_state = self.correction_net(correction_input)
            
        return {
            'safety_flag': safety_flag,
            'reflection': reflection,
            'error_prob': error_prob,
            'corrected_state': corrected_state,
            'needs_reflection': error_prob > 0.5
        }
        
    def get_reflection_history(self):
        """Get history of past reflections"""
        return self.reflection_memory
        
    def reflect_on_error(self, error_context):
        """Generate targeted reflection based on error context"""
        if not self.reflection_memory:
            return None
            
        # Combine error context with past reflections
        past_reflections = torch.stack(self.reflection_memory)
        avg_reflection = past_reflections.mean(dim=0)
        
        # Generate new reflection considering error context
        combined = torch.cat([avg_reflection, error_context], dim=-1)
        new_reflection = self.reflection_net(combined)
        
        return new_reflection

class Value(nn.Module):
    """
    Value  for assigning safety values to different memory tokens or hidden states.
    """
    def __init__(self, token_dim):
        super(Value, self).__init__()
        self.token_dim = token_dim

        # Assign safety values to tokens
        self.value_layer = nn.Linear(token_dim, 1)

    def forward(self, tokens):
        # Compute safety values
        values = torch.sigmoid(self.value_layer(tokens))
        return values

class MemoryAugmentedTransformer(nn.Module):
    """
    Transformer model augmented with PFC, Metacognitive, and Value  modules for safety regulation.
    """
    def __init__(self, transformer, hidden_dim, memory_dim, context_dim, config):
        super(MemoryAugmentedTransformer, self).__init__()
        self.transformer = transformer
        self.pfc = PFCModule(hidden_dim, memory_dim, context_dim)
        self.metacognitive = MetacognitiveModule(hidden_dim, memory_dim)
        self.value_ = Value(memory_dim)
        self.config = config

    def forward(self, hidden_states, memory, context, goal_embedding):
        # Pass through transformer
        transformer_output = self.transformer(hidden_states)

        # PFC module processing
        modulated_output, inhibitory_signals = self.pfc(transformer_output, memory, context)

        # Monitor for safety and generate reflection
        metacog_output = self.metacognitive(modulated_output, memory)
        safety_flag = metacog_output['safety_flag']
        reflection = metacog_output['reflection']
        error_prob = metacog_output['error_prob']
        corrected_state = metacog_output['corrected_state']
        needs_reflection = metacog_output['needs_reflection']

        # Evaluate safety values for memory tokens
        memory_values = self.value_(memory)

        # Get subgoal importance from memory
        subgoal_importance = self.config.model.goal_manager.get_subgoal_importance(goal_embedding)

        # Decide whether to use corrected state based on error probability
        if needs_reflection:
            final_output = corrected_state
        else:
            final_output = modulated_output

        return {
            'output': final_output,
            'safety_flag': safety_flag,
            'inhibitory_signals': inhibitory_signals,
            'memory_values': memory_values,
            'reflection': reflection,
            'error_prob': error_prob,
            'subgoal_importance': subgoal_importance
        }

class BinaryLatentMemoryPool:
    """Enhanced memory pool for storing and managing binary latent states with improved memory management"""
    def __init__(self, pool_size: int, latent_dim: int, device: str = 'cuda',
                 memory_decay: float = 0.99, importance_threshold: float = 0.1,
                 compression_ratio: float = 0.5, diversity_threshold: float = 0.3,
                 initial_temperature: float = 1.0, initial_exploration: float = 0.1,
                 min_temperature: float = 0.1, max_temperature: float = 2.0,
                 temperature_decay: float = 0.99, exploration_decay: float = 0.995,
                 n_star: int = 4):  # Target number of correct responses per query for balance score
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.device = device
        self.memory_states = torch.zeros(pool_size, latent_dim).to(device)
        self.binary_states = torch.zeros(pool_size, latent_dim).bool().to(device)
        self.state_importance = torch.zeros(pool_size).to(device)
        self.memory_age = torch.zeros(pool_size).to(device)
        self.memory_decay = memory_decay
        self.importance_threshold = importance_threshold
        self.compression_ratio = compression_ratio
        self.diversity_threshold = diversity_threshold
        
        # B* temperature and exploration parameters
        self.temperature = initial_temperature
        self.exploration_rate = initial_exploration
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature_decay = temperature_decay
        self.exploration_decay = exploration_decay
        
        # B-STAR monitoring
        self.n_star = n_star  # Target number of correct responses for balance score
        self.temperature_history = []
        self.exploration_history = []
        self.balance_scores = []
        self.exploration_scores = []  # Track Pass@K-S
        self.exploitation_scores = []  # Track Reward@K-S
        
        # Track access frequency for each memory state
        self.access_count = torch.zeros(pool_size).to(device)
        self.last_access = torch.zeros(pool_size).to(device)
        
        # Binary state encoder/decoder
        self.state_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        ).to(device)
        
        # Enhanced memory usage and compression statistics
        self.usage_stats = {
            'updates': 0,
            'states_added': 0,
            'states_dropped': 0,
            'importance_history': [],
            'memory_age_history': [],
            'compression_ratio_history': [],
            'binary_sparsity_history': [],
            'reconstruction_error_history': [],
            'diversity_scores': [],
            'access_patterns': [],
            'memory_lifetime': [],
            'importance_distribution': []
        }
        
    def compute_balance_score(self, n_correct: int, n_selected: int) -> float:
        """Compute B-STAR balance score for current batch"""
        # Discount factor encouraging sufficient correct responses
        discount = min(n_correct / self.n_star, 1.0)
        # Ratio of correct responses among selected
        ratio = n_correct / max(n_selected, 1)
        # Balance score combining quantity and quality
        return float(discount * ratio)

    def update(self, new_states: torch.Tensor, k: int, binary_latents: Optional[torch.Tensor] = None, 
              update_params: bool = True, correct_mask: Optional[torch.Tensor] = None):
        """Update memory pool with enhanced importance scoring and diversity selection"""
        with torch.no_grad():
            # Update memory age and access patterns
            self.memory_age += 1
            current_step = self.usage_stats['updates']
            self.last_access = torch.where(
                self.access_count > 0,
                current_step - self.last_access,
                self.memory_age
            )

            # Apply temperature scaling and exploration
            if binary_latents is not None:
                new_binary_states = binary_latents
            else:
                # Encode new states to binary with temperature scaling
                binary_probs = self.state_encoder(new_states)
                
                # Apply temperature scaling
                binary_probs = torch.sigmoid(torch.log(binary_probs + 1e-10) / self.temperature)
                
                # Apply exploration
                if torch.rand(1).item() < self.exploration_rate:
                    # Random exploration
                    new_binary_states = torch.rand_like(binary_probs) < self.exploration_rate
                else:
                    # Greedy selection with temperature
                    new_binary_states = (binary_probs > 0.5).bool()

            # Compute binary entropy for importance
            binary_entropy = -torch.mean(
                new_binary_states.float() * torch.log2(new_binary_states.float() + 1e-10) +
                (1 - new_binary_states.float()) * torch.log2(1 - new_binary_states.float() + 1e-10),
                dim=1
            )
            
            # Enhanced importance scoring combining multiple factors
            recency_score = 1.0 / (1.0 + self.memory_age)
            access_score = self.access_count / (self.usage_stats['updates'] + 1)
            l2_norm = torch.norm(self.memory_states, dim=1)
            content_score = l2_norm / (torch.max(l2_norm) + 1e-8)
            
            # Compute exponential decay
            time_decay = self.memory_decay ** self.memory_age
            
            # Combine scores with learned weights
            self.state_importance = (
                0.4 * recency_score + 
                0.3 * access_score +
                0.3 * content_score
            ) * time_decay
            
            # Calculate importance scores combining binary entropy, information content and recency
            state_entropy = self._compute_state_entropy(self.memory_states)
            binary_importance = binary_entropy / binary_entropy.max()  # Normalize to [0,1]
            recency_weight = 1.0 / (1.0 + self.memory_age)
            
            # Combine scores with learned weights
            self.state_importance = (
                0.4 * binary_importance +
                0.3 * state_entropy * recency_weight +
                0.3 * (1.0 / (1.0 + self.memory_age))  # Pure recency score
            )
            
            # Filter out low importance states
            valid_mask = self.state_importance > self.importance_threshold
            valid_states = self.memory_states[valid_mask]
            valid_binary = self.binary_states[valid_mask]
            valid_importance = self.state_importance[valid_mask]
            valid_age = self.memory_age[valid_mask]
            
            # Keep most important states
            if len(valid_states) > self.pool_size - k:
                _, indices = torch.topk(valid_importance, self.pool_size - k)
                kept_states = valid_states[indices]
                kept_binary = valid_binary[indices]
                kept_age = valid_age[indices]
            else:
                kept_states = valid_states
                kept_binary = valid_binary
                kept_age = valid_age
            
            # Process new states with enhanced diversity selection
            if new_states.size(0) > k:
                # Compute pairwise cosine similarity
                similarities = torch.nn.functional.cosine_similarity(
                    new_states.unsqueeze(1),
                    new_states.unsqueeze(0),
                    dim=2
                )
                
                # Greedy diversity maximization
                selected_indices = []
                available_indices = set(range(len(new_states)))
                
                # Start with highest importance state
                importance = torch.norm(new_states, dim=1)
                first_idx = importance.argmax().item()
                selected_indices.append(first_idx)
                available_indices.remove(first_idx)
                
                while len(selected_indices) < k and available_indices:
                    # Compute maximum similarity to selected states
                    max_similarities = similarities[list(available_indices)][:, selected_indices].max(dim=1)[0]
                    
                    # Select state with lowest maximum similarity
                    next_idx = min(available_indices, key=lambda i: max_similarities[i].item())
                    
                    # Only add if diversity threshold is met
                    if max_similarities[next_idx].item() < self.diversity_threshold:
                        selected_indices.append(next_idx)
                    available_indices.remove(next_idx)
                
                selected_indices = torch.tensor(selected_indices, device=self.device)
                new_states = new_states[selected_indices]
                new_binary_states = new_binary_states[selected_indices]
            
            # Concatenate and update
            self.memory_states = torch.cat([kept_states, new_states], dim=0)
            self.binary_states = torch.cat([kept_binary, new_binary_states], dim=0)
            self.memory_age = torch.cat([
                kept_age,
                torch.zeros(len(new_states), device=self.device)
            ])
            
            # Ensure pool size stays constant
            if self.memory_states.size(0) > self.pool_size:
                self.memory_states = self.memory_states[:self.pool_size]
                self.binary_states = self.binary_states[:self.pool_size]
                self.memory_age = self.memory_age[:self.pool_size]
            
            # Compute compression metrics
            compression_ratio = self._compute_compression_ratio()
            reconstruction_error = self._compute_reconstruction_error()
            binary_sparsity = self._compute_binary_sparsity()
            
            # Update enhanced statistics
            self.usage_stats['updates'] += 1
            self.usage_stats['states_added'] += len(new_states)
            self.usage_stats['states_dropped'] += (len(valid_states) - len(kept_states))
            
            # Compute B-STAR metrics and update parameters
            if update_params:
                if correct_mask is not None:
                    # Get number of correct and selected responses
                    n_correct = correct_mask.sum().item()
                    n_selected = len(new_states)
                    
                    # Compute balance score
                    balance_score = self.compute_balance_score(n_correct, n_selected)
                    self.balance_scores.append(balance_score)
                    
                    # Track exploration (Pass@K-S)
                    exploration_score = n_correct / max(k, 1)  # Ratio of correct responses
                    self.exploration_scores.append(exploration_score)
                    
                    # Track exploitation (Reward@K-S) 
                    exploitation_score = n_correct / max(n_selected, 1)  # Quality of selection
                    self.exploitation_scores.append(exploitation_score)
                    
                    # Update temperature and exploration based on balance score
                    self._update_temperature_and_exploration(balance_score)
                else:
                    # Fallback to original update if no correct_mask provided
                    self._update_temperature_and_exploration()
                
                # Track history
                self.temperature_history.append(self.temperature)
                self.exploration_history.append(self.exploration_rate)
            
            # Track detailed memory statistics
            self.usage_stats['importance_history'].append(self.state_importance.mean().item())
            self.usage_stats['memory_age_history'].append(self.memory_age.mean().item())
            self.usage_stats['compression_ratio_history'].append(compression_ratio)
            self.usage_stats['binary_sparsity_history'].append(binary_sparsity)
            self.usage_stats['reconstruction_error_history'].append(reconstruction_error)
            
            # Track diversity and memory lifetime metrics
            if len(new_states) > 1:
                diversity_score = 1.0 - torch.nn.functional.cosine_similarity(
                    new_states.unsqueeze(1),
                    new_states.unsqueeze(0),
                    dim=2
                ).mean().item()
                self.usage_stats['diversity_scores'].append(diversity_score)
            
            self.usage_stats['access_patterns'].append(self.access_count.mean().item())
            self.usage_stats['memory_lifetime'].append(
                (self.memory_age * (self.state_importance > self.importance_threshold).float()).mean().item()
            )
            self.usage_stats['importance_distribution'].append(
                self.state_importance.histc(bins=10, min=0, max=1).tolist()
            )
            
    def get_states(self) -> torch.Tensor:
        """Get current memory states with importance weighting and binary reconstruction"""
        # Weight states by importance
        weights = torch.softmax(self.state_importance, dim=0)
        weighted_states = self.memory_states * weights.unsqueeze(1)
        
        # Reconstruct from binary states when beneficial
        binary_states = self.binary_states.float()
        reconstructed_states = self.state_decoder(binary_states)
        
        # Use binary reconstruction when compression ratio is good
        use_binary = self._compute_compression_ratio() < self.compression_ratio
        return torch.where(use_binary.unsqueeze(1), reconstructed_states, weighted_states)
    
    def _select_diverse_binary_states(self, binary_states: torch.Tensor, k: int) -> torch.Tensor:
        """Select diverse states using Hamming distance between binary representations"""
        if len(binary_states) <= k:
            return torch.arange(len(binary_states))
            
        # Compute pairwise Hamming distances
        distances = torch.cdist(
            binary_states.float(),
            binary_states.float(),
            p=0  # Hamming distance
        )
        
        # Greedy selection of diverse states
        selected = [0]  # Start with first state
        while len(selected) < k:
            # Compute minimum distance to selected states
            min_dist = distances[selected].min(dim=0)[0]
            
            # Select state with maximum minimum distance
            remaining = list(set(range(len(binary_states))) - set(selected))
            next_idx = max(remaining, key=lambda i: min_dist[i])
            selected.append(next_idx)
            
        return torch.tensor(selected, device=binary_states.device)
    
    def _compute_state_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute entropy of states as importance measure"""
        # Normalize states to probability distribution
        probs = torch.softmax(states, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy
        
    def _compute_compression_ratio(self) -> float:
        """Compute effective compression ratio of binary states"""
        binary_size = self.binary_states.numel() / 8  # Convert bits to bytes
        full_size = self.memory_states.numel() * self.memory_states.element_size()
        return binary_size / full_size
        
    def _compute_reconstruction_error(self) -> float:
        """Compute reconstruction error of binary states"""
        with torch.no_grad():
            binary_states = self.binary_states.float()
            reconstructed = self.state_decoder(binary_states)
            error = nn.MSELoss()(reconstructed, self.memory_states)
            return error.item()
            
    def _compute_binary_sparsity(self) -> float:
        """Compute sparsity of binary states"""
        return 1.0 - (self.binary_states.float().mean().item())
        
    def _update_temperature_and_exploration(self, balance_score: Optional[float] = None):
        """Update temperature and exploration rate based on B-STAR balance score"""
        if balance_score is not None:
            # Adjust temperature based on balance score
            if balance_score < 0.5:  # Poor balance
                # Increase temperature to encourage exploration
                self.temperature = min(
                    self.max_temperature,
                    self.temperature / self.temperature_decay
                )
            else:  # Good balance
                # Gradually reduce temperature
                self.temperature = max(
                    self.min_temperature,
                    self.temperature * self.temperature_decay
                )
            # Adjust exploration rate based on balance score
            if balance_score < 0.3:  # Increase exploration significantly
                self.exploration_rate = min(1.0, self.exploration_rate / (self.exploration_decay * 0.8))
            elif balance_score < 0.7:  # Moderate balance
                # Increase exploration moderately
                self.exploration_rate = min(1.0, self.exploration_rate / self.exploration_decay)
            else:  # Good balance
                # Reduce exploration gradually
                self.exploration_rate *= self.exploration_decay
        else:
            # Fallback to original update logic
            # Decay temperature
            self.temperature = max(
                self.min_temperature,
                self.temperature * self.temperature_decay
            )
            
            # Increase temperature if memory performance is poor
            avg_importance = self.state_importance.mean().item()
            if avg_importance < self.importance_threshold:
                self.temperature = min(
                    self.max_temperature,
                    self.temperature / self.temperature_decay
                )
            
            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay
            
            # Increase exploration if memory is too homogeneous
            if self._compute_memory_diversity() < self.diversity_threshold:
                self.exploration_rate = min(1.0, self.exploration_rate / self.exploration_decay)
    
    def _compute_memory_diversity(self) -> float:
        """Compute diversity of memory states"""
        if len(self.memory_states) <= 1:
            return 0.0
            
        # Compute pairwise cosine similarities
        normalized = torch.nn.functional.normalize(self.memory_states, dim=1)
        similarities = torch.mm(normalized, normalized.t())
        
        # Average similarity (lower means more diverse)
        avg_similarity = (similarities.sum() - similarities.diag().sum()) / (similarities.numel() - similarities.size(0))
        
        # Convert to diversity score (1 - similarity)
        return 1.0 - avg_similarity.item()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage and compression statistics"""
        stats = {
            'pool_size': self.pool_size,
            'current_size': len(self.memory_states),
            'mean_importance': self.state_importance.mean().item(),
            'mean_age': self.memory_age.mean().item(),
            'compression_ratio': self._compute_compression_ratio(),
            'binary_sparsity': self._compute_binary_sparsity(),
            'reconstruction_error': self._compute_reconstruction_error(),
            'updates': self.usage_stats['updates'],
            'total_states_added': self.usage_stats['states_added'],
            'total_states_dropped': self.usage_stats['states_dropped'],
            'importance_history': self.usage_stats['importance_history'],
            'age_history': self.usage_stats['memory_age_history'],
            'compression_history': self.usage_stats['compression_ratio_history'],
            'sparsity_history': self.usage_stats['binary_sparsity_history'],
            'reconstruction_history': self.usage_stats['reconstruction_error_history'],
            
            # B-STAR specific stats
            'temperature': self.temperature,
            'exploration_rate': self.exploration_rate,
            'temperature_history': self.temperature_history,
            'exploration_history': self.exploration_history,
            'memory_diversity': self._compute_memory_diversity(),
            
            # B-STAR monitoring metrics
            'balance_scores': self.balance_scores,
            'exploration_scores': self.exploration_scores,
            'exploitation_scores': self.exploitation_scores,
            'mean_balance_score': sum(self.balance_scores) / max(len(self.balance_scores), 1),
            'mean_exploration_score': sum(self.exploration_scores) / max(len(self.exploration_scores), 1),
            'mean_exploitation_score': sum(self.exploitation_scores) / max(len(self.exploitation_scores), 1)
        }
        return stats

class MultiStateRNN(nn.Module):
    """Multi-state RNN with memory pool integration"""
    def __init__(self, hidden_size: int, num_layers: int, memory_size: int = 1024, k_tokens: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k_tokens = k_tokens
        
        # RNN cells for each layer
        self.cells = nn.ModuleList([
            nn.LSTMCell(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Memory pool
        self.memory_pool = BinaryLatentMemoryPool(memory_size, hidden_size) #Replaced Memory Pool with Binary Latent Memory Pool
        
        # Memory integration
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # State compression policy
        self.compression_enabled = False
        self.max_states = None
        
    def forward(self, x: torch.Tensor, states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with memory integration and state compression
        Args:
            x: Input tensor [batch_size, hidden_size]
            states: Optional list of (h, c) states for each layer
        Returns:
            output: Output tensor [batch_size, hidden_size]
            new_states: Updated states for each layer
        """
        batch_size = x.size(0)
        
        # Initialize states if not provided
        if states is None:
            states = [(torch.zeros(batch_size, self.hidden_size, device=x.device),
                      torch.zeros(batch_size, self.hidden_size, device=x.device))
                     for _ in range(self.num_layers)]
        
        # Get memory tokens using k-select from the memory pool
        memory_tokens = self.memory_pool.get_states().unsqueeze(0).expand(batch_size, -1, -1)

        
        # Process through layers
        current_input = x
        new_states = []
        for i, (h, c) in enumerate(states):
            # Concatenate input with memory tokens
            combined_input = torch.cat([current_input.unsqueeze(1), memory_tokens], dim=1)
            
            # Apply memory attention
            attended_memory, _ = self.memory_attention(
                current_input.unsqueeze(1),
                memory_tokens,
                memory_tokens
            )
            
            # Combine with current input
            enhanced_input = current_input + attended_memory.squeeze(1)
            
            # RNN cell forward pass
            new_h, new_c = self.cells[i](enhanced_input, (h, c))
            
            # Apply compression if enabled
            if self.compression_enabled and self.max_states is not None:
                new_h, new_c = self._compress_states(new_h, new_c)
                
            new_states.append((new_h, new_c))
            current_input = new_h
            
        # Update memory pool with last K tokens
        self.memory_pool.update(current_input[-self.k_tokens:], self.k_tokens)
            
        return current_input, new_states
        
    def _compress_states(self, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress states if they exceed max_states"""
        if h.size(1) > self.max_states:
            # Keep most important states based on activation magnitude
            importance = torch.norm(h, dim=2)  # [batch_size, num_states]
            _, indices = torch.topk(importance, self.max_states, dim=1)
            h = torch.gather(h, 1, indices.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            c = torch.gather(c, 1, indices.unsqueeze(-1).expand(-1, -1, c.size(-1)))
        return h, c

class GoalNode:
    """Node in the goal tree representing a subgoal"""
    def __init__(self, text: str, parent=None):
        self.text = text
        self.parent = parent
        self.children = []
        self.importance = 1.0
        self.visits = 0
        self.rewards = []
        
    def add_child(self, child_text: str) -> 'GoalNode':
        """Add a child node with given text"""
        child = GoalNode(child_text, self)
        self.children.append(child)
        return child
        
    def update(self, reward: float):
        """Update node statistics with new reward"""
        self.visits += 1
        self.rewards.append(reward)
        self.importance = np.mean(self.rewards)
        
    def is_leaf(self) -> bool:
        """Check if node is"""
        return len(self.children) == 0
        
    def get_path(self) -> List[str]:
        """Get the path from root to this node"""
        path = [self.text]
        current = self
        while current.parent:
            current = current.parent
            path.insert(0, current.text)
        return path

#class GoalManager was removed because this is already present in the AI's thinking processes. 
   
class ChainOfThoughtReward(nn.Module):
    """
    Reward model for Chain-of-Thought (CoT) reasoning that integrates with MCTS.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Confidence scoring 
        self.confidence_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # Value prediction for MCTS
        self.value_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Policy head for MCTS action selection
        self.policy_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.action_space_size)
        )
        
        # Temperature parameter for confidence scoring
        self.temperature = 0.1
        
    def compute_confidence_score(self, token_logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence score for each token using top-5 alternatives"""
        # Get top 5 probabilities
        top_probs = F.softmax(token_logits, dim=-1).topk(5, dim=-1).values
        
        # Compute confidence as ratio of top probability to sum of top 5
        confidence = top_probs[:, :, 0] / (top_probs.sum(dim=-1) + 1e-10)
        
        return confidence
        
    def compute_reward(self, reasoning_path: torch.Tensor) -> torch.Tensor:
        """Compute reward for a reasoning path based on confidence and value prediction"""
        # Get confidence scores for each step
        confidence_scores = self.confidence_net(reasoning_path)
        
        # Get value prediction
        value = self.value_net(reasoning_path.mean(dim=1))
        
        # Combine confidence and value prediction
        reward = confidence_scores.mean(dim=1) * (value + 1) / 2  # Scale value to [0,1]
        
        return reward
        
    def get_mcts_outputs(self, state: torch.Tensor) -> tuple:
        """Get policy logits and value prediction for MCTS"""
        policy_logits = self.policy_net(state)
        value = self.value_net(state).squeeze(-1)
        
        return policy_logits, value
        
    def update_temperature(self, reward_history: list):
        """Adaptively update temperature based on reward history"""
        if len(reward_history) < 10:
            return
            
        # Compute mean and std of recent rewards
        recent_rewards = torch.tensor(reward_history[-10:])
        mean_reward = recent_rewards.mean()
        std_reward = recent_rewards.std()
        
        # Adjust temperature based on reward statistics
        if mean_reward > 0.8:  # High rewards - reduce temperature
            self.temperature = max(0.05, self.temperature * 0.95)
        elif mean_reward < 0.4:  # Low rewards - increase temperature
            self.temperature = min(1.0, self.temperature * 1.05)
        elif std_reward < 0.1:  # Low variance - increase temperature
            self.temperature = min(1.0, self.temperature * 1.02)

    def forward(self, reasoning_states: torch.Tensor, actions: torch.Tensor = None) -> dict:
        """Forward pass computing rewards and MCTS outputs"""
        # Compute rewards
        rewards = self.compute_reward(reasoning_states)
        
        # Get MCTS outputs from final state
        policy_logits, values = self.get_mcts_outputs(reasoning_states[:, -1])
        
        outputs = {
            'rewards': rewards,
            'policy_logits': policy_logits,
            'values': values
        }
        
        # Compute policy loss if actions provided
        if actions is not None:
            policy_loss = F.cross_entropy(policy_logits, actions)
            outputs['policy_loss'] = policy_loss
            
        return outputs

class FactualityRewardModel(nn.Module):
    """Reward model for factuality assessment"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Factuality projection
        self.factuality_projection = nn.Linear(config.d_model, config.d_model)
        
        # Factuality attention for scoring
        self.factuality_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Temperature for scaling scores
        self.factuality_temperature = 0.1
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute factuality scores"""
        # Project to factuality space
        factuality_hidden = self.factuality_projection(hidden_states)
        
        # Compute attention-based factuality scores
        scores, _ = self.factuality_attention(
            factuality_hidden, factuality_hidden, factuality_hidden
        )
        
        # Average across heads and apply temperature scaling
        scores = scores.mean(dim=1)  # [batch_size, seq_len]
        scores = torch.sigmoid(scores / self.factuality_temperature)
        
        return scores

class GoalManager:
    """
    Manages hierarchical goal decomposition and selection using an internal approach.
    """
    def __init__(self, config):
        self.config = config
        
        # Goal tree
        self.root = None
        
        # Tracking
        self.selected_goals = []
        self.goal_history = []
        self.goal_embeddings = {}  # Store embeddings of goals for similarity checks
        
        # Parameters
        self.max_goals = config.max_goals
        self.min_importance = config.min_importance
        self.exploration_factor = config.exploration_factor
        
        # B-STAR parameters
        self.temperature = self.config.initial_temperature  # Initial temperature for exploration
        self.min_temperature = 0.1 # Min temperature
        self.temperature_decay = 0.99 # Decay for temperature
        self.similarity_threshold = 0.85 # Threshold for filtering similar goals
        
    def initialize_tree(self, main_goal: str):
        """Initialize goal tree with main goal"""
        self.root = GoalNode(main_goal)
        # Generate and store embedding for the main goal
        self.goal_embeddings[main_goal] = self.generate_embedding(main_goal)

    def decompose_goal(self, goal_node: GoalNode, context: dict) -> List[str]:
        """
        Decompose a goal into subgoals based on context using the model's internal subgoal generation module.
        """
        if not hasattr(self.config, "model"):
            raise AttributeError("The config object must have a 'model' attribute referring to the ByteLatentTransformer model.")

        model = self.config.model

        # Ensure the model is in evaluation mode
        model.eval()

        with torch.no_grad():
            # Encode the current goal and context into embeddings
            goal_embedding = model._encode_goals([goal_node.text])
            context_embedding = model._encode_context(context)

            # Generate subgoal embeddings
            subgoal_embeddings = model._generate_subgoals(goal_embedding, context_embedding)

            # Filter out similar subgoals
            unique_subgoal_embeddings = model._filter_subgoals(subgoal_embeddings)

            # Decode subgoal embeddings into text
            subgoals = [model._decode_goal(embedding) for embedding in unique_subgoal_embeddings]

            # Add new subgoals as children and store their embeddings
            for subgoal in subgoals:
                goal_node.add_child(subgoal)
                self.goal_embeddings[subgoal] = self.generate_embedding(subgoal)  # Assuming you add this method to GoalManager

        return subgoals

    def generate_embedding(self, text: str) -> torch.Tensor:
        """
        Generates an embedding for a given text using the model's encoder.
        Assumes that the model has a method 'encode_text' that returns an embedding.
        """
        if not hasattr(self.config, "model"):
            raise AttributeError("The config object must have a 'model' attribute referring to the ByteLatentTransformer model.")

        model = self.config.model
        model.eval()  # Ensure the model is in evaluation mode

        with torch.no_grad():
            # Tokenize the text (assuming your tokenizer can handle single strings)
            tokens = self.config.tokenizer.tokenize(text)
            # Convert tokens to tensor (add batch dimension)
            input_tensor = {k: v.unsqueeze(0).to(self.config.device) for k, v in tokens.items()}
            # Get the model's output
            outputs = model(input_tensor)
            # Extract the embedding (e.g., from the encoder states)
            # This assumes that the last encoder state is a good representation of the whole sequence
            embedding = outputs['encoder_states'][-1].mean(dim=1).squeeze(0)

        return embedding
        
    def _filter_similar(self, subgoals: List[str]) -> List[str]:
        """Filter out similar subgoals based on their embeddings."""
        if not subgoals:
            return []

        # Retrieve embeddings for each subgoal
        embeddings = [self.goal_embeddings[subgoal] for subgoal in subgoals]

        # Compute cosine similarity between all pairs of embeddings
        similarity_matrix = torch.nn.functional.cosine_similarity(
            torch.stack(embeddings).unsqueeze(1), 
            torch.stack(embeddings).unsqueeze(0), 
            dim=2
        )

        # Filter out subgoals that are too similar to others
        unique_subgoals = []
        for i, subgoal in enumerate(subgoals):
            # Check if the subgoal is not too similar to any previously added unique subgoals
            if all(similarity_matrix[i, j] < self.similarity_threshold for j in range(i)):
                unique_subgoals.append(subgoal)

        return unique_subgoals
        
    def select_goals(self, context: dict) -> List[str]:
      """Select most relevant goals based on current context, using B-STAR principles"""
      selected = []
    
      # Get all leaf nodes
      leaves = self._get_leaves(self.root)
    
      if not leaves:
          return []
    
      # Score leaves based on context and exploration-exploitation balance
      scores = self._score_goals(leaves, context)
    
      # Apply temperature-based sampling (similar to B-STAR)
      # Higher temperature -> more exploration
      # Lower temperature -> more exploitation (favoring higher-scoring goals)
      if self.temperature > 0:
          exp_scores = np.exp(np.array(scores) / self.temperature)
          probs = exp_scores / np.sum(exp_scores)
    
          # Select goals based on the probabilities
          num_goals_to_select = min(self.max_goals, len(leaves))
          selected_indices = np.random.choice(
              len(leaves),
              size=num_goals_to_select,
              replace=False,
              p=probs
          )
          selected = [leaves[i].text for i in selected_indices]
      else:
          # If temperature is 0 or less, perform greedy selection (exploitation only)
          k = min(self.max_goals, len(leaves))
          selected_indices = np.argpartition(scores, -k)[-k:]
          selected = [leaves[i].text for i in selected_indices]
    
      self.selected_goals = selected
      self.goal_history.append(selected)
    
      return selected

    def update_tree(self, reward: float):
        """Update tree statistics based on reward, and adjust temperature using B-STAR principles"""
        # Update selected nodes
        for goal in self.selected_goals:
            node = self._find_node(self.root, goal)
            if node:
                node.update(reward)

        # Update temperature based on success/failure
        # If recent rewards are high, decrease temperature to encourage exploitation
        # If recent rewards are low, increase temperature to encourage exploration
        if len(self.goal_history) > 10: # Consider recent 10 steps for temperature update
            recent_rewards = [np.mean(self._find_node(self.root, g).rewards) for g in self.goal_history[-10:] if self._find_node(self.root, g)]

            if len(recent_rewards) > 0:
                avg_recent_reward = np.mean(recent_rewards)
                if avg_recent_reward > 0.7: # Assume 0.7 as a threshold for success
                    self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
                elif avg_recent_reward < 0.4:
                    self.temperature = min(self.config.initial_temperature, self.temperature / self.temperature_decay)

        # Prune low importance nodes
        self._prune_tree(self.root)

        ```python
        
    def _score_goals(self, goal_nodes: List[GoalNode], context: dict) -> List[float]:
        """Score goals based on context using an LLM, importance, and exploration factor."""
        scores = []
        for node in goal_nodes:
            # Retrieve the subgoal embedding
            subgoal_embedding = self.goal_embeddings.get(node.text)

            if subgoal_embedding is None:
                print(f"Warning: Embedding not found for subgoal '{node.text}'. Skipping.")
                continue  # Skip this subgoal if embedding is not found

            # Encode the context into an embedding
            context_embedding = self.config.model._encode_context(context)

            # Calculate the relevance score using the model's internal scoring module
            relevance_score = self.config.model._score_subgoals([subgoal_embedding], context_embedding)[0]

            # Encourage exploration of less-visited nodes
            exploration_bonus = self.exploration_factor / (node.visits + 1)

            # Overall score is a combination of relevance, importance, and exploration
            score = (relevance_score * node.importance) + exploration_bonus
            scores.append(score)

        return scores

    def get_subgoal_importance(self, subgoal_embedding: torch.Tensor) -> float:
        """
        Retrieves the importance of a subgoal based on its embedding.
        This is a placeholder method that you'll need to implement based on how you store
        and update subgoal importance in the GoalManager.
        """
        # This is a placeholder. You might want to:
        # 1. Maintain a dictionary mapping subgoal text to importance.
        # 2. Use the memory pool to store subgoal embeddings and their importance scores.
        # 3. Implement a mechanism to query this dictionary/memory pool based on the subgoal embedding.

        # For now, let's assume a simple dictionary lookup:
        subgoal_text = self._find_subgoal_text(subgoal_embedding)
        if subgoal_text:
            node = self._find_node(self.root, subgoal_text)
            if node:
                return node.importance
        return 1.0  # Default importance if not found
        
    def _find_subgoal_text(self, subgoal_embedding: torch.Tensor) -> Optional[str]:
        """
        Finds the text of a subgoal based on its embedding using the model's decoder.
        """
        self.model.eval()  # Ensure the model is in evaluation mode

        with torch.no_grad():
            # 1. Prepare Decoder Input:
            # Start with an SOS token (if your model uses it).
            sos_token_id = self.config.tokenizer.bos_token_id if hasattr(self.config.tokenizer, 'bos_token_id') else 0  # Replace 0 with your model's SOS token ID if it's different
            decoder_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=self.config.device)

            # 2. Initialize Decoder Hidden State:
            # Use the subgoal_embedding to initialize the decoder's hidden state.
            # You may need to adjust the dimensions based on your decoder's architecture.
            decoder_hidden = subgoal_embedding.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            if self.config.decoder_layers > 1:
                decoder_hidden = decoder_hidden.repeat(self.config.decoder_layers, 1, 1)

            # 3. Decode with Beam Search:
            max_length = 50  # Maximum length of the decoded subgoal (adjust as needed)
            beam_width = 5   # Adjust as needed
            sequences = [(decoder_input, 0.0, decoder_hidden)]  # (sequence, score, hidden_state)

            for _ in range(max_length):
                all_candidates = []
                for seq, score, hidden in sequences:
                    # Stop if the sequence has already generated an EOS token
                    if seq[0, -1].item() == self.config.tokenizer.eos_token_id:
                        all_candidates.append((seq, score, hidden))
                        continue

                    # Get the decoder output
                    decoder_output, new_hidden = self.model.local_decoder['transformer'](seq, memory=None, hidden=hidden)

                    # Get the log probabilities of the next tokens
                    log_probs = F.log_softmax(self.model.local_decoder['output'](decoder_output), dim=-1)

                    # Select the top-k candidates
                    top_k_log_probs, top_k_indices = log_probs[0, -1, :].topk(beam_width)

                    # Create new candidate sequences
                    for i in range(beam_width):
                        token_log_prob = top_k_log_probs[i].item()
                        token_index = top_k_indices[i].unsqueeze(0).unsqueeze(0)
                        new_seq = torch.cat([seq, token_index], dim=-1)
                        new_score = score + token_log_prob
                        all_candidates.append((new_seq, new_score, new_hidden))

                # Select the top-k candidates overall
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                sequences = ordered[:beam_width]

            # Select the best sequence
            best_sequence, _, _ = sequences[0]

            # 4. Convert to Text:
            # Convert the best sequence (token IDs) back to text using the tokenizer.
            subgoal_text = self.config.tokenizer.decode(best_sequence.squeeze(0).tolist())
            
            # Remove SOS and EOS tokens from the decoded goal
            subgoal_text = subgoal_text.replace(self.config.tokenizer.convert_ids_to_tokens(self.config.tokenizer.bos_token_id), "")  # Remove SOS token
            subgoal_text = subgoal_text.replace(self.config.tokenizer.convert_ids_to_tokens(self.config.tokenizer.eos_token_id), "")  # Remove EOS token

            return subgoal_text.strip()
        
    def _get_leaves(self, node: GoalNode) -> List[GoalNode]:
        """Recursively get all leaf nodes"""
        if node.is_leaf():
            return [node]
        else:
            leaves = []
            for child in node.children:
                leaves.extend(self._get_leaves(child))
            return leaves
            
    def _find_node(self, root: GoalNode, goal_text: str) -> Optional[GoalNode]:
        """Find a node with given text in the tree"""
        if root.text == goal_text:
            return root
        for child in root.children:
            found = self._find_node(child, goal_text)
            if found:
                return found
        return None
        
    def _prune_tree(self, node: GoalNode):
        """Recursively prune low importance nodes"""
        node.children = [
            child for child in node.children
            if child.importance >= self.min_importance
        ]
        for child in node.children:
            self._prune_tree(child)

class ByteLatentTransformer(nn.Module):
    """BLT model with enhanced binary latent memory integration, brain-inspired components, episodic memory, Dynamic patching, multi-state RNN integration, and SELFGOAL."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        vocab_size: int = 256,  # Byte-level vocabulary
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        mem_pool_size: int = 100,
        latent_dim: int = 64,
        device: str = 'cuda',
        memory_decay: float = 0.99,
        importance_threshold: float = 0.1,
        compression_ratio: float = 0.5,
        diversity_threshold: float = 0.3,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
        temperature_decay: float = 0.99,
        initial_exploration: float = 0.1,
        exploration_decay: float = 0.995,
        n_star: int = 4,
        episodic_mem_capacity: int = 1000,
        episodic_mem_config: Optional[EpisodicMemoryConfig] = None,
        retrieval_k: int = 5,
        use_retrieval: bool = True,
        initial_consolidation_interval: int = 1000,
        min_consolidation_interval: int = 100,
        max_consolidation_interval: int = 5000,
        consolidation_adaptation_rate: float = 0.1,
        memory_integration_method: str = 'attention',
        episodic_transfer_rate: float = 0.2,
        transfer_similarity_threshold: float = 0.8,
        importance_update_factor: float = 0.5,
        novelty_factor: float = 0.5,  # Weight for novelty in importance calculation
        relevance_factor: float = 0.3,  # Weight for relevance in importance calculation
        reward_factor: float = 0.2,  # Weight for reward in importance calculation
        importance_threshold_episodic: float = 0.2,  # Minimum importance for an experience to be considered for transfer
        prediction_error_factor: float = 0.4,  # Weight for prediction error in surprise calculation
        learning_progress_factor: float = 0.6  # Weight for learning progress in surprise calculation

    ):
        super().__init__()

        self.d_model = d_model
        self.device = device
        self.use_retrieval = use_retrieval
        self.retrieval_k = retrieval_k
        self.initial_consolidation_interval = initial_consolidation_interval
        self.min_consolidation_interval = min_consolidation_interval
        self.max_consolidation_interval = max_consolidation_interval
        self.consolidation_adaptation_rate = consolidation_adaptation_rate
        self.consolidation_interval = initial_consolidation_interval # Start with the initial interval
        self.memory_integration_method = memory_integration_method
        self.consolidation_step_count = 0
        self.episodic_transfer_rate = episodic_transfer_rate
        self.transfer_similarity_threshold = transfer_similarity_threshold
        self.importance_update_factor = importance_update_factor
        self.novelty_factor = novelty_factor
        self.relevance_factor = relevance_factor
        self.reward_factor = reward_factor
        self.importance_threshold_episodic = importance_threshold_episodic
        self.prediction_error_factor = prediction_error_factor
        self.learning_progress_factor = learning_progress_factor

        # Embedding layer for byte-level inputs
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Binary latent memory module
        self.memory_pool = BinaryLatentMemoryPool(
            pool_size=mem_pool_size,
            latent_dim=latent_dim,
            device=device,
            memory_decay=memory_decay,
            importance_threshold=importance_threshold,
            compression_ratio=compression_ratio,
            diversity_threshold=diversity_threshold,
            initial_temperature=initial_temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            temperature_decay=temperature_decay,
            initial_exploration=initial_exploration,
            exploration_decay=exploration_decay,
            n_star=n_star
        )

        # Output layer to produce logits over vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Episodic Memory
        if episodic_mem_config is None:
            episodic_mem_config = EpisodicMemoryConfig(capacity=episodic_mem_capacity)
        self.episodic_memory = HierarchicalEpisodicMemory(
            capacity=episodic_mem_capacity,
            config=episodic_mem_config
        )

        # Memory Integration Mechanisms
        if self.memory_integration_method == 'attention':
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
        elif self.memory_integration_method == 'gating':
            self.memory_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        # Consolidation thread and flag
        self.consolidation_thread = None
        self.stop_consolidation_thread = False

        activation_dict = {}  # Dictionary to store activations
   
        # User Profile Generation Module
        self.user_profile_generator = nn.LSTM(config.d_model, config.user_profile_dim, num_layers=1, batch_first=True) # Example: LSTM

        # User Profile Storage (Option 1: In-memory dictionary)
        self.user_profiles = {}  # {user_id: user_profile_embedding}

        # Add a linear layer to project combined embeddings to the desired dimension
        self.context_projection = nn.Linear(config.d_model * 3 + config.user_profile_dim, config.d_model)

def calculate_importance(self, experience: Dict, context_embedding: torch.Tensor) -> float:
        """
        Calculates the importance score for an experience in episodic memory.

        Args:
            experience: A dictionary representing the experience.
            context_embedding: The embedding of the context associated with the experience.

        Returns:
            The importance score (float).
        """

        # 1. Novelty (example: based on cluster assignment or distance to cluster centroids)
        cluster_id = self.get_cluster_for_experience(experience)
        if cluster_id is not None:
            cluster_centroid = self.episodic_memory.clusters[cluster_id]['centroid']
            novelty_score = 1.0 - F.cosine_similarity(context_embedding, cluster_centroid.unsqueeze(0)).item()
        else:
            # Handle cases where the experience doesn't belong to any cluster yet
            novelty_score = 1.0  # Assign maximum novelty

        # 2. Relevance (example: based on similarity to current goal or context)
        if 'goal' in experience:
          relevance_score = F.cosine_similarity(
              context_embedding,
              experience['goal'].unsqueeze(0),
              dim=1
          ).item()
        else:
          relevance_score = 0.0 # Provide default value if the goal is not available

        # 3. Reward (example: based on the final reward associated with the experience)
        reward_score = experience.get('final_reward', 0.0)  # Get reward from experience, default to 0

        # 4. Surprise (example: based on prediction error or learning progress)
        surprise_score = self.calculate_surprise(experience)

        # Combine the factors with weighted averaging
        importance_score = (
            self.novelty_factor * novelty_score +
            self.relevance_factor * relevance_score +
            self.reward_factor * reward_score +
            (1 - self.novelty_factor - self.relevance_factor - self.reward_factor) * surprise_score
        )

        return importance_score

  def calculate_importance(self, experience: Dict, context_embedding: torch.Tensor) -> float:
        """
        Calculates the importance score for an experience in episodic memory.

        Args:
            experience: A dictionary representing the experience.
            context_embedding: The embedding of the context associated with the experience.

        Returns:
            The importance score (float).
        """

        # 1. Novelty (example: based on cluster assignment or distance to cluster centroids)
        cluster_id = self.get_cluster_for_experience(experience)
        if cluster_id is not None:
            cluster_centroid = self.episodic_memory.clusters[cluster_id]['centroid']
            novelty_score = 1.0 - F.cosine_similarity(context_embedding, cluster_centroid.unsqueeze(0)).item()
        else:
            # Handle cases where the experience doesn't belong to any cluster yet
            novelty_score = 1.0  # Assign maximum novelty

        # 2. Relevance (example: based on similarity to current goal or context)
        if 'goal' in experience:
          relevance_score = F.cosine_similarity(
              context_embedding,
              experience['goal'].unsqueeze(0),
              dim=1
          ).item()
        else:
          relevance_score = 0.0 # Provide default value if the goal is not available

        # 3. Reward (example: based on the final reward associated with the experience)
        reward_score = experience.get('final_reward', 0.0)  # Get reward from experience, default to 0

        # 4. Surprise (example: based on prediction error or learning progress)
        surprise_score = self.calculate_surprise(experience)

        # Combine the factors with weighted averaging
        importance_score = (
            self.novelty_factor * novelty_score +
            self.relevance_factor * relevance_score +
            self.reward_factor * reward_score +
            (1 - self.novelty_factor - self.relevance_factor - self.reward_factor) * surprise_score
        )

        return importance_score

    def get_cluster_for_experience(self, experience: Dict) -> Optional[int]:
      """
      Finds the cluster ID to which an experience belongs (if any).

      Args:
          experience: A dictionary representing the experience.

      Returns:
          The cluster ID (int) or None if the experience doesn't belong to any cluster.
      """
      goal_context_embedding = torch.cat([experience['goal'], experience['context']], dim=-1).cpu().detach().numpy()

      if self.episodic_memory.cluster_method == 'hdbscan':
          cluster_id = hdbscan.prediction.approximate_predict(self.episodic_memory.clusterer, [goal_context_embedding])[0][0]
          return cluster_id if cluster_id != -1 else None
      elif self.episodic_memory.cluster_method == 'som':
          bmu = self.episodic_memory.som.winner(goal_context_embedding)
          cluster_id = str(bmu)
          return cluster_id if cluster_id in self.episodic_memory.clusters else None
      else:  # Assuming kmeans
          min_dist = float('inf')
          closest_cluster_id = None
          for cluster_id, cluster_data in self.episodic_memory.clusters.items():
              dist = torch.norm(torch.tensor(goal_context_embedding) - cluster_data['centroid'])
              if dist < min_dist:
                  min_dist = dist
                  closest_cluster_id = cluster_id
          return closest_cluster_id

    def calculate_surprise(self, experience: Dict) -> float:
        """
        Calculates a surprise score for an experience based on prediction error and learning progress.

        Args:
            experience: A dictionary representing the experience.

        Returns:
            The surprise score (float).
        """

        # 1. Prediction Error (example: using a simple difference between expected and actual reward)
        if 'expected_reward' in experience and 'final_reward' in experience:
            prediction_error = abs(experience['expected_reward'] - experience['final_reward'])
        else:
            prediction_error = 0.0

        # 2. Learning Progress (example: based on how much the model's parameters changed after the experience)
        # This requires tracking parameter changes during training, which might be complex to implement directly here.
        # As a placeholder, you could use a proxy like the number of times the experience has been used for training.
        learning_progress = experience.get('training_count', 0.0)  # You'll need to update this count during training

        # Normalize the factors (you might need to adjust the normalization based on your specific task)
        normalized_prediction_error = prediction_error / (prediction_error + 1.0)  # Simple normalization to [0, 1]
        normalized_learning_progress = math.log(1 + learning_progress)

        # Combine factors using weighted averaging
        surprise_score = (
            self.prediction_error_factor * normalized_prediction_error +
            self.learning_progress_factor * normalized_learning_progress
        )

        return surprise_score
    
    def store_experience(self, experience: Dict):
      """Stores an experience in the episodic memory, along with an importance score."""
      context_embedding = experience['context']  # Assuming you have a way to get this

      # Calculate importance
      importance_score = self.calculate_importance(experience, context_embedding)

      # Add importance to the experience dictionary
      experience['importance'] = importance_score

      self.episodic_memory.push(experience)

        def get_activation(name):
            def hook(model, input, output):
            activation_dict[name] = output.detach()
            return hook

        # Register forward hooks for specific layers
        self.local_encoder['transformer'].layers[0].register_forward_hook(get_activation('encoder_layer_0'))
        self.region_embeddings['visual'].register_forward_hook(get_activation('visual_region'))
        # ... (register hooks for other layers/regions) #Need to change this to work with region dictionary in progam. 

        # Initialize SELFGOAL components
        self.goal_manager = GoalManager(config)
        self.cot_reward = ChainOfThoughtReward(config)
        
        # Initialize binary latent memory pool
        self.memory_pool = BinaryLatentMemoryPool(
            pool_size=config.memory_size,
            latent_dim=config.d_model,
            device=config.device,
            memory_decay=0.99,
            importance_threshold=0.1,
            compression_ratio=0.5,
            diversity_threshold=0.3
        )

        # Add parameters for MCTS
        self.mcts_rollouts = config.initial_mcts_rollouts
        self.mcts_expansion_factor = config.mcts_expansion_factor
        self.mcts_c = config.mcts_c

        # Subgoal Generation Module
        self.subgoal_generator = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # Project combined state to subgoal space
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),  # Generate subgoal embedding
        )
        self.fc_step = nn.Linear(1, config.d_model)
        self.context_projection = nn.Linear(config.d_model * 3, config.d_model)

        # Subgoal Filtering Module
        self.subgoal_filter = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # Combine subgoal embeddings
            nn.ReLU(),
            nn.Linear(config.d_model, 1),  # Output similarity score
            nn.Sigmoid()  # Scale to [0, 1]
        )

        # Subgoal Scoring Module
        self.subgoal_scorer = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # Combine subgoal and context
            nn.ReLU(),
            nn.Linear(config.d_model, 1),  # Output relevance score
            nn.Sigmoid()  # Scale to [0, 1]
        )

    def set_mcts_params(self, rollouts: int, expansion_factor: int, c: float):
        """Sets the parameters for MCTS."""
        self.mcts_rollouts = rollouts
        self.mcts_expansion_factor = expansion_factor
        self.mcts_c = c

    def set_temperature(self, temperature: float):
      """Sets the temperature for goal selection in the GoalManager."""
      self.goal_manager.temperature = temperature
    
    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'goal_manager_state_dict': self.goal_manager.get_state_dict(),
            'memory_pool_state_dict': self.memory_pool.get_state_dict()
        }
        
        save_path = self.config.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint - modified to load goal manager and memory pool"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Load GoalManager state
        self.goal_manager.load_state_dict(checkpoint['goal_manager_state_dict'])

        # Load MemoryPool state
        self.memory_pool.load_state_dict(checkpoint['memory_pool_state_dict'])

    def _generate_subgoals(self, goal_embedding: torch.Tensor, context_embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Generates potential subgoals based on the current goal and context embeddings.
        """
        # Combine goal and context embeddings as input to the subgoal generator
        combined_embedding = torch.cat([goal_embedding, context_embedding], dim=-1)

        # Generate a set of subgoal embeddings
        num_subgoals = self.config.max_goals  # You can adjust the number of subgoals to generate
        subgoal_embeddings = self.subgoal_generator(combined_embedding.unsqueeze(0).repeat(num_subgoals, 1))

        return subgoal_embeddings

    def _filter_subgoals(self, subgoal_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
      """
      Filters out redundant or highly similar subgoals based on their embeddings.
      """
      unique_subgoals = []
      for i in range(subgoal_embeddings.size(0)):
          is_unique = True
          for j in range(i):
              # Concatenate two subgoal embeddings for similarity comparison
              combined = torch.cat([subgoal_embeddings[i], subgoal_embeddings[j]], dim=-1)
              # Get a similarity score from the filter module
              similarity_score = self.subgoal_filter(combined.unsqueeze(0)).squeeze()

              # If similarity is above threshold, consider them non-unique
              if similarity_score.item() > self.config.similarity_threshold:
                  is_unique = False
                  break
          
          if is_unique:
              unique_subgoals.append(subgoal_embeddings[i])

      return unique_subgoals

    def _score_subgoals(self, subgoal_embeddings: List[torch.Tensor], context_embedding: torch.Tensor) -> List[float]:
      """
      Scores subgoals based on their relevance to the current context and their importance.
      """
      scores = []
      for subgoal_embedding in subgoal_embeddings:
          # Combine subgoal and context embeddings for relevance scoring
          combined = torch.cat([subgoal_embedding, context_embedding], dim=-1)
          # Get a relevance score from the scoring module
          relevance_score = self.subgoal_scorer(combined.unsqueeze(0)).squeeze()

          # Retrieve the importance of the subgoal from the goal manager (if available)
          # This is a placeholder; you'll need to implement a mechanism to track and retrieve
          # the importance of subgoals in the GoalManager, potentially using the memory pool.
          importance = self.goal_manager.get_subgoal_importance(subgoal_embedding) or 1.0

          # Combine relevance and importance to get the final score
          score = relevance_score.item() * importance
          scores.append(score)

      return scores

    def _update_memory_with_goals(self, goal_embeddings: List[torch.Tensor], scores: List[float]):
      """Updates the memory pool with new goal embeddings and their scores."""
      if not goal_embeddings:
          return

      # Convert goal embeddings and scores to tensors
      goal_embeddings_tensor = torch.stack(goal_embeddings)
      scores_tensor = torch.tensor(scores, device=self.config.device)

      # Update the binary latent memory pool with the new goal embeddings
      # The binary representation will be computed internally by the memory pool
      self.memory_pool.update(goal_embeddings_tensor, k=len(goal_embeddings), update_params=True, scores=scores_tensor)

    def _decompose_and_select_subgoals(self, main_goal: str, context: dict) -> List[str]:
        """
        Decomposes the main goal into a hierarchy of subgoals and selects the most relevant ones.
        """
        # Initialize the goal tree with the main goal
        self.goal_manager.initialize_tree(main_goal)

        # Encode the main goal and context into embeddings
        goal_embedding = self._encode_goals([main_goal])
        context_embedding = self._encode_context(context)

        # Recursively decompose the main goal into a hierarchy of subgoals
        def decompose_recursively(node: GoalNode, goal_embedding: torch.Tensor, context_embedding: torch.Tensor, level: int = 0, max_levels: int = 3):
            if level >= max_levels:
                return

            # Generate, filter, and score subgoals for the current node
            subgoal_embeddings = self._generate_subgoals(goal_embedding, context_embedding)
            unique_subgoal_embeddings = self._filter_subgoals(subgoal_embeddings)
            subgoal_scores = self._score_subgoals(unique_subgoal_embeddings, context_embedding)

            # Update memory with new subgoals and their scores
            self._update_memory_with_goals(unique_subgoal_embeddings, subgoal_scores)

            # Decode subgoal embeddings into text for creating GoalNode objects
            subgoals_text = [self._decode_goal(embedding) for embedding in unique_subgoal_embeddings]

            # Add the generated subgoals as children of the current node
            for subgoal_text in subgoals_text:
                child_node = node.add_child(subgoal_text)
                # Recursively decompose each child subgoal
                decompose_recursively(child_node, goal_embedding, context_embedding, level + 1, max_levels)

        decompose_recursively(self.goal_manager.root, goal_embedding, context_embedding)

        # Select the most relevant subgoals based on the current context using the GoalManager
        selected_subgoals = self.goal_manager.select_goals(context)

        return selected_subgoals 
        
    def _extract_high_level_goal(self, x: torch.Tensor) -> str:
        """Extract high-level goal from input context using byte-level analysis"""
        # Convert bytes to text for analysis
        text = ''.join([chr(b.item()) for b in x.flatten() if 0 <= b.item() < 128])
        
        # Look for goal-related keywords and patterns
        goal_indicators = [
            'goal:', 'objective:', 'task:', 'aim:', 'purpose:',
            'achieve', 'accomplish', 'complete', 'solve', 'optimize'
        ]
        
        # Find the most relevant sentence containing goal information
        sentences = text.split('.')
        goal_sentence = None
        max_indicators = 0
        
        for sent in sentences:
            n_indicators = sum(1 for ind in goal_indicators if ind.lower() in sent.lower())
            if n_indicators > max_indicators:
                max_indicators = n_indicators
                goal_sentence = sent
                
        if goal_sentence is None:
            # Default to first sentence if no clear goal indicators
            goal_sentence = sentences[0] if sentences else "Process and analyze input data"
            
        return goal_sentence.strip()
        
    def _encode_goals(self, goals: List[str]) -> torch.Tensor:
        """Encode list of goals into embedding space"""
        # Convert goals to byte sequences
        goal_bytes = []
        for goal in goals:
            bytes_tensor = torch.tensor([ord(c) for c in goal], device=self.config.device)
            goal_bytes.append(bytes_tensor)
            
        # Pad sequences to same length
        max_len = max(len(g) for g in goal_bytes)
        padded_goals = torch.zeros(len(goals), max_len, dtype=torch.long, device=self.config.device)
        for i, g in enumerate(goal_bytes):
            padded_goals[i, :len(g)] = g
            
        # Get embeddings using the byte embeddings
        goal_embeds = self.local_encoder['embedding'](padded_goals)
        
        # Pool embeddings (mean pooling)
        goal_embeds = goal_embeds.mean(dim=1)  # [num_goals, d_model]
        
        # Combine goal embeddings with attention
        if len(goals) > 1:
            # Self-attention over goals
            attn_weights = torch.matmul(goal_embeds, goal_embeds.transpose(-2, -1))
            attn_weights = torch.softmax(attn_weights / np.sqrt(goal_embeds.size(-1)), dim=-1)
            goal_context = torch.matmul(attn_weights, goal_embeds)
            goal_context = goal_context.mean(dim=0)  # [d_model]
        else:
            goal_context = goal_embeds.squeeze(0)  # [d_model]
            
        return goal_context
        
    def _decompose_and_select_subgoals(self, main_goal: str, context: dict) -> List[str]:
      """
      Decomposes the main goal into a hierarchy of subgoals and selects the most relevant ones.

      Args:
          main_goal: The main goal to be decomposed.
          context: A dictionary containing contextual information that can guide the decomposition and selection.

      Returns:
          A list of selected subgoals that are most relevant to the current context and the main goal.
      """
      # Initialize the goal tree with the main goal
      self.goal_manager.initialize_tree(main_goal)

      # Recursively decompose the main goal into a hierarchy of subgoals
      def decompose_recursively(node: GoalNode, level: int = 0, max_levels: int = 3):
          if level >= max_levels:
              return
          
          # Generate subgoals for the current node
          subgoals = self.goal_manager.decompose_goal(node, context)
          
          # Add the generated subgoals as children of the current node
          for subgoal in subgoals:
              child_node = node.add_child(subgoal)
              # Recursively decompose each child subgoal
              decompose_recursively(child_node, level + 1, max_levels)

      decompose_recursively(self.goal_manager.root)

      # Select the most relevant subgoals based on the current context
      selected_subgoals = self.goal_manager.select_goals(context)

      return selected_subgoals

def generate_user_profile(self, interaction_history: List[Dict], goal_embedding: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generates or updates a user profile embedding based on the interaction history.
        """
        # 1. Convert interaction history to embeddings:
        interaction_embeddings = []
        for interaction in interaction_history:
            # Assuming each interaction is a dictionary with 'user_input' and 'model_response'
            user_input_embed = self.local_encoder['embedding'](interaction['user_input']).mean(dim=1)
            model_response_embed = self.local_encoder['embedding'](interaction['model_response']).mean(dim=1)
            interaction_embeddings.append(torch.cat([user_input_embed, model_response_embed], dim=-1))

        interaction_embeddings = torch.stack(interaction_embeddings)  # [seq_len, embedding_dim]

        # 2. Process with User Profile Generator (e.g., LSTM):
        _, (hidden_state, _) = self.user_profile_generator(interaction_embeddings.unsqueeze(0))  # Add batch dimension
        user_profile_embedding = hidden_state[-1, :, :]  # Get the final hidden state

        return user_profile_embedding.squeeze(0)  # Remove batch dimension

    def update_user_profile(self, user_id: str, interaction_history: List[Dict], goal_embedding: torch.Tensor, context_embedding: torch.Tensor):
        """
        Updates the stored user profile for a given user ID.
        """
        user_profile_embedding = self.generate_user_profile(interaction_history, goal_embedding, context_embedding)
        self.user_profiles[user_id] = user_profile_embedding

    def get_user_profile(self, user_id: str) -> Optional[torch.Tensor]:
        """
        Retrieves the stored user profile for a given user ID.
        """
        return self.user_profiles.get(user_id)

    def _execute_with_chain_of_thought(self, subgoals: List[str], initial_input: torch.Tensor, user_id: Optional[str] = None) -> Tuple[torch.Tensor, List[Dict]]:
      """
      Executes a sequence of subgoals using a Chain-of-Thought approach, where each subgoal
      builds upon the results of the previous one. Includes self-reflection and backtracking.
      Now includes user_id to handle user-specific interactions.
      """
      current_output = initial_input
      reasoning_steps = []

      # Add context and memory to initial input
      combined_input = initial_input
      if hasattr(self, 'memory_pool'):
          memory_states = self.memory_pool.get_states()
          if memory_states.numel() > 0:
              combined_input = torch.cat([combined_input, memory_states], dim=0)

      # Retrieve or create user profile embedding
      if user_id is not None:
        user_profile_embedding = self.get_user_profile(user_id)
        if user_profile_embedding is None:
            user_profile_embedding = self.generate_user_profile(
                interaction_history=[],  # Start with an empty history for new users
                goal_embedding=self._encode_goals([subgoals[0]]),  # Encode the first subgoal as an initial goal
                context_embedding=self._encode_context({'input': initial_input})  # Encode the initial input as context
            )
            self.update_user_profile(user_id, [], self._encode_goals([subgoals[0]]), self._encode_context({'input': initial_input}))
      else:
          user_profile_embedding = None

      for subgoal in subgoals:
          # Update the current goal for this step
          self.current_goal = subgoal

          # Prepare the context for this subgoal
          context = {
              'input': combined_input,
              'step': len(reasoning_steps) + 1
          }

          # Add user profile embedding to the context if available
          if user_profile_embedding is not None:
              context['user_profile'] = user_profile_embedding

          # Execute the current subgoal
          subgoal_output = self.forward(combined_input, goal_context=self._encode_goals([subgoal]), context_embedding=self._encode_context(context))

          # Update the current output with the result of the subgoal execution
          current_output = subgoal_output['hidden_states']

          # Track the reasoning step for CoT reward computation
          reasoning_steps.append({
              'type': 'subgoal_execution',
              'goal': subgoal,
              'input': combined_input.detach(),
              'output': subgoal_output['hidden_states'].detach()
          })

          # Compute confidence scores for this step
          with torch.no_grad():
              logits = self.local_decoder['output'](subgoal_output['hidden_states'])
              confidence = self.cot_reward.compute_confidence_score(logits)
              reasoning_steps[-1]['confidence'] = confidence.mean().item()

          # Reflection step (if needed)
          if hasattr(self, 'metacognitive'):
            reflection_needed = self.metacognitive(current_output, memory_states)['needs_reflection']
            if reflection_needed:
                reflection = self.metacognitive(current_output, memory_states)['reflection']
                if len(reasoning_steps) > 1:
                    previous_step = reasoning_steps.pop()
                    current_output = previous_step['input']
                    reasoning_steps.append({
                        'type': 'reflection',
                        'reflection': reflection.detach(),
                        'previous_output': previous_step['output']
                    })

          # Update combined input for the next subgoal
          combined_input = current_output
          if hasattr(self, 'memory_pool'):
              memory_states = self.memory_pool.get_states()
              if memory_states.numel() > 0:
                  combined_input = torch.cat([combined_input, memory_states], dim=0)

          # Update user profile based on the interaction (if a user ID is provided)
          if user_id is not None:
              self.update_user_profile(user_id, reasoning_steps, self._encode_goals([subgoal]), self._encode_context(context))

      # Compute the final CoT reward based on the entire reasoning path
      final_reward = self.cot_reward.compute_reward(torch.stack([step['output'] for step in reasoning_steps if 'output' in step]))

      return current_output, reasoning_steps, final_reward

    def forward(self, x: torch.Tensor, goal_context: Optional[torch.Tensor] = None,
            brain_regions: Optional[Dict[str, torch.Tensor]] = None,
            rnn_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            sensory_data: Optional[torch.Tensor] = None,
            tactile_data: Optional[torch.Tensor] = None,
            audio_data: Optional[torch.Tensor] = None,
            moral_principles: Optional[List[str]] = None,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            memory_states: Optional[torch.Tensor] = None,
            correct_mask: Optional[torch.Tensor] = None
            user_profile: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
             ) -> torch.Tensor:
        """
        Forward pass of the ByteLatentTransformer.

        Args:
            src: Input sequence (batch_size, seq_len)
            src_mask: Mask for the source sequence (optional)
            src_key_padding_mask: Padding mask for the source sequence (optional)
            memory_states: Latent memory states from previous interactions (optional)

        Returns:
            Output logits over the vocabulary (batch_size, seq_len, vocab_size)
        """

        """Forward pass implementing BLT workflow with RNN processing, brain region integration, and SELFGOAL"""
        # Initialize SELFGOAL for this forward pass if not already initialized
        if not hasattr(self, 'current_goal'):
            # Extract high-level goal from input context
            self.current_goal = self._extract_high_level_goal(x)
            self.goal_manager.initialize_tree(self.current_goal)
            
        # Get current context for SELFGOAL
        context = {
            'input': x,
            'brain_regions': brain_regions,
            'rnn_states': rnn_states,
            'step': getattr(self, 'step_counter', 0)
        }
        
        # Select relevant subgoals using SELFGOAL
        selected_goals = self.goal_manager.select_goals(context)

          # 1. Embed the input tokens
        src_embed = self.embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)

        # 2. Encode the input sequence
        encoder_output = self.encoder(
            src_embed,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, d_model)

        # 3. Optionally incorporate memory states from short term memory
        if memory_states is not None:
            # Combine encoder output with memory states (e.g., concatenation or attention)
            # This is a placeholder for a more sophisticated memory integration mechanism
            encoder_output = torch.cat([encoder_output, memory_states.unsqueeze(1)], dim=1)

        # 4. Optionally retrieve from episodic memory and incorporate
        if self.use_retrieval and hasattr(self, 'episodic_memory'):
            # Assuming the last hidden state of the encoder output can represent the context
            context = encoder_output[:, -1, :]

            # Create a dummy goal for demonstration purposes; adapt as needed
            dummy_goal = torch.randn_like(context)
            retrieved_memories = self.retrieve_memories(dummy_goal, context, self.retrieval_k)

            if retrieved_memories:
                # Filter memories based on importance
                important_memories = [
                    m for m in retrieved_memories if m['importance'] > self.importance_threshold_episodic
                ]

                if important_memories:
                    # Process retrieved memories using attention or gating
                    retrieved_embeddings = torch.stack([
                        torch.cat([m['goal'], m['context']], dim=-1) for m in important_memories
                    ])  # (retrieval_k, goal_dim + context_dim)

                    if self.memory_integration_method == 'attention':
                        # Add a query vector for the current context
                        query = context.unsqueeze(1)  # (batch_size, 1, d_model)

                        # Reshape retrieved_embeddings for attention
                        retrieved_embeddings = retrieved_embeddings.unsqueeze(0).expand(
                            context.size(0), -1, -1
                        )  # (batch_size, retrieval_k, goal_dim + context_dim)

                        # Use multi-head attention to get a weighted combination of memories
                        attn_output, _ = self.memory_attention(
                            query, retrieved_embeddings, retrieved_embeddings
                        )  # (batch_size, 1, d_model)

                        # Combine with the current context
                        combined_context = attn_output.squeeze(1)  # (batch_size, d_model)

                    elif self.memory_integration_method == 'gating':
                        # Concatenate context and mean of retrieved memories
                        mean_retrieved = retrieved_embeddings.mean(dim=0, keepdim=True)  # (1, goal_dim + context_dim)
                        
                        # Expand mean_retrieved to match the batch size of context
                        mean_retrieved = mean_retrieved.expand(context.size(0), -1) # (batch_size, goal_dim + context_dim)

                        gate_input = torch.cat([context, mean_retrieved], dim=-1)  # (batch_size, d_model + goal_dim + context_dim)

                        # Pass through the gating mechanism
                        gate = self.memory_gate(gate_input)  # (batch_size, d_model)

                        # Combine with the current context using the gate
                        combined_context = gate * context + (1 - gate) * mean_retrieved[:, :self.d_model]  # Assuming goal_dim + context_dim includes d_model

                else:
                    combined_context = context
            else:
                combined_context = context
        else:
            combined_context = encoder_output[:, -1, :]

        # 5. Update memory pool (for subsequent interactions)
        self.memory_pool.update(
            new_states=encoder_output,
            k=10,  # You can adjust k as needed
            update_params=True,  # Only update if needed
            correct_mask=correct_mask  # Pass in feedback signal
        )

        # 6. Generate output logits
        output = self.fc_out(combined_context)  # (batch_size, vocab_size)

        # Update consolidation step count and trigger if needed
        self.consolidation_step_count += 1
        if self.consolidation_step_count >= self.consolidation_interval:
            self.consolidate_memories()
            self.consolidation_step_count = 0

        return output

    def start_consolidation_thread(self):
        """Starts the background consolidation thread."""
        if self.consolidation_thread is None or not self.consolidation_thread.is_alive():
            self.stop_consolidation_thread = False
            self.consolidation_thread = threading.Thread(target=self.background_consolidation)
            self.consolidation_thread.daemon = True
            self.consolidation_thread.start()

    def stop_consolidation_thread(self):
        """Stops the background consolidation thread."""
        self.stop_consolidation_thread = True
        if self.consolidation_thread is not None and self.consolidation_thread.is_alive():
            self.consolidation_thread.join()

    def background_consolidation(self):
        """Performs memory consolidation in the background."""
        while not self.stop_consolidation_thread:
            # You can adjust the sleep time as needed
            time.sleep(self.consolidation_interval)
            if not self.stop_consolidation_thread:
                self.consolidate_memories()

    def adaptive_consolidation_interval(self):
        """
        Adjusts the consolidation interval based on factors like:
        - Memory usage
        - Performance metrics (e.g., loss, accuracy)
        - Other relevant factors (e.g., stage of training)
        """

        # Example: Adjust based on memory pool usage and performance
        memory_stats = self.memory_pool.get_stats()
        current_usage = memory_stats['current_size'] / memory_stats['pool_size']
        
        # Example performance metric (e.g., get training loss from a global variable or callback)
        # Replace this with your actual performance tracking mechanism
        try:
          current_performance = self.get_latest_training_loss() # Assume you have a method to track this
        except:
          current_performance = 0.1 # Provide a default or fallback value

        # If memory usage is high and performance is poor, reduce the interval (more frequent consolidation)
        if current_usage > 0.8 and current_performance > 0.2: # Example thresholds
            self.consolidation_interval = max(
                self.min_consolidation_interval,
                int(self.consolidation_interval * (1 - self.consolidation_adaptation_rate))
            )
        # If memory usage is low and performance is good, increase the interval (less frequent consolidation)
        elif current_usage < 0.5 and current_performance < 0.1: # Example thresholds
            self.consolidation_interval = min(
                self.max_consolidation_interval,
                int(self.consolidation_interval * (1 + self.consolidation_adaptation_rate))
            )
        
        # Add other factors and adjustments as needed

        print(f"Consolidation interval updated to: {self.consolidation_interval}")

    def transfer_from_episodic_memory(self):
      """
      Transfers relevant experiences from episodic memory to the memory pool based on
      similarity to current memory pool states.
      """
      if not self.episodic_memory:
          return

      num_to_transfer = int(len(self.episodic_memory) * self.episodic_transfer_rate)
      if num_to_transfer == 0:
        return

      # Get a representative sample of embeddings from the memory pool
      memory_pool_samples = self.memory_pool.get_states()

      # Get a set of candidate experiences from episodic memory for transfer consideration
      candidate_experiences = self.episodic_memory.get_k_nearest_memories(
          query_embedding=memory_pool_samples.mean(dim=0),  # Use mean of memory pool samples as query
          k=num_to_transfer
      )

      experiences_to_transfer = []
      for experience in candidate_experiences:
          # Compute similarity between the experience embedding and memory pool samples
          experience_embedding = torch.cat([experience['goal'], experience['context']], dim=-1)

          similarity_scores = F.cosine_similarity(
              experience_embedding.unsqueeze(0),
              memory_pool_samples,
              dim=1
          )
          max_similarity = similarity_scores.max().item()

          # Transfer if similarity is above threshold and importance is high enough
          if max_similarity > self.transfer_similarity_threshold and experience['importance'] > self.importance_threshold_episodic:
              experiences_to_transfer.append(experience)

      # Update the memory pool with transferred experiences
      for experience in experiences_to_transfer:
          self.update_memory_pool_from_episodic(experience)

    def update_memory_pool_from_episodic(self, experience: Dict):
        """
        Updates the memory pool with an experience from episodic memory.

        Args:
            experience: A dictionary representing the experience from episodic memory.
                        Must have keys 'goal' and 'context' with corresponding tensor values.
        """

        # Convert experience to a suitable format for the memory pool
        # (This is highly dependent on your specific use case)
        # For example, you might use the concatenated goal and context as the state:
        episodic_state = torch.cat([experience['goal'], experience['context']], dim=-1)

        # 1. Find a similar state in the memory pool (if any)
        memory_states = self.memory_pool.get_states()
        similarity_scores = F.cosine_similarity(episodic_state.unsqueeze(0), memory_states, dim=1)
        most_similar_index = similarity_scores.argmax().item()
        max_similarity = similarity_scores[most_similar_index].item()

        # 2. Decide whether to replace the similar state or add a new one
        if max_similarity > self.transfer_similarity_threshold:
            # Replace the similar state if the new state has higher importance (or other criteria)
            # Example: Update importance based on episodic memory (you'll need a mechanism to assign importance in episodic memory)
            episodic_importance = experience.get('importance', 1.0)  # Get importance from episodic memory or default to 1.0

            current_importance = self.memory_pool.state_importance[most_similar_index].item()

            # Update importance with a weighted average (or other combination)
            updated_importance = (1 - self.importance_update_factor) * current_importance + self.importance_update_factor * episodic_importance

            if updated_importance > current_importance:
              self.memory_pool.memory_states[most_similar_index] = episodic_state
              self.memory_pool.state_importance[most_similar_index] = updated_importance
              # Update other relevant attributes (e.g., age, access count) if needed
              self.memory_pool.memory_age[most_similar_index] = 0  # Reset age for transferred state
        else:
            # Add as a new state if below capacity, otherwise, it might replace a low-importance state based on memory_pool's logic
            self.memory_pool.update(new_states=episodic_state.unsqueeze(0), k=1)

    def consolidate_memories(self):
        """
        Performs offline memory consolidation.
        This can include:
        - Re-clustering experiences in episodic memory.
        - Transferring consolidated experiences from episodic memory to the memory pool.
        - Adaptively adjusting the consolidation interval.
        - Other optimization or compression operations on the memory.
        """
        print("Consolidating memories...")

        # Re-cluster experiences in episodic memory
        self.episodic_memory.perform_cluster_maintenance()

        # Transfer relevant experiences from episodic memory to memory pool
        self.transfer_from_episodic_memory()

        # Adaptively adjust the consolidation interval
        self.adaptive_consolidation_interval()

        print("Memory consolidation complete.")

            # Create a dummy goal for demonstration purposes; adapt as needed
            dummy_goal = torch.randn_like(context)
            retrieved_memories = self.retrieve_memories(dummy_goal, context, self.retrieval_k)

            if retrieved_memories:
                # Process retrieved memories using attention or gating
                retrieved_embeddings = torch.stack([
                    torch.cat([m['goal'], m['context']], dim=-1) for m in retrieved_memories
                ])  # (retrieval_k, goal_dim + context_dim)

                if self.memory_integration_method == 'attention':
                    # Add a query vector for the current context
                    query = context.unsqueeze(1)  # (batch_size, 1, d_model)

                    # Reshape retrieved_embeddings for attention
                    retrieved_embeddings = retrieved_embeddings.unsqueeze(0).expand(
                        context.size(0), -1, -1
                    )  # (batch_size, retrieval_k, goal_dim + context_dim)

                    # Use multi-head attention to get a weighted combination of memories
                    attn_output, _ = self.memory_attention(
                        query, retrieved_embeddings, retrieved_embeddings
                    )  # (batch_size, 1, d_model)

                    # Combine with the current context
                    combined_context = attn_output.squeeze(1)  # (batch_size, d_model)

                elif self.memory_integration_method == 'gating':
                    # Concatenate context and mean of retrieved memories
                    mean_retrieved = retrieved_embeddings.mean(dim=0, keepdim=True)  # (1, goal_dim + context_dim)
                    
                    # Expand mean_retrieved to match the batch size of context
                    mean_retrieved = mean_retrieved.expand(context.size(0), -1) # (batch_size, goal_dim + context_dim)

                    gate_input = torch.cat([context, mean_retrieved], dim=-1)  # (batch_size, d_model + goal_dim + context_dim)

                    # Pass through the gating mechanism
                    gate = self.memory_gate(gate_input)  # (batch_size, d_model)

                    # Combine with the current context using the gate
                    combined_context = gate * context + (1 - gate) * mean_retrieved[:, :self.d_model]  # Assuming goal_dim + context_dim includes d_model

            else:
                combined_context = context
        else:
            combined_context = encoder_output[:, -1, :]

        # 5. Update memory pool (for subsequent interactions)
        self.memory_pool.update(
            new_states=encoder_output,
            k=10,  # You can adjust k as needed
            update_params=True,  # Only update if needed
            correct_mask=correct_mask  # Pass in feedback signal
        )

        # 6. Generate output logits
        output = self.fc_out(combined_context)  # (batch_size, vocab_size)

        # Update consolidation step count and trigger if needed
        self.consolidation_step_count += 1
        if self.consolidation_step_count >= self.consolidation_interval:
            self.consolidate_memories()
            self.consolidation_step_count = 0

        return output

    def start_consolidation_thread(self):
        """Starts the background consolidation thread."""
        if self.consolidation_thread is None or not self.consolidation_thread.is_alive():
            self.stop_consolidation_thread = False
            self.consolidation_thread = threading.Thread(target=self.background_consolidation)
            self.consolidation_thread.daemon = True
            self.consolidation_thread.start()

    def stop_consolidation_thread(self):
        """Stops the background consolidation thread."""
        self.stop_consolidation_thread = True
        if self.consolidation_thread is not None and self.consolidation_thread.is_alive():
            self.consolidation_thread.join()

    def background_consolidation(self):
        """Performs memory consolidation in the background."""
        while not self.stop_consolidation_thread:
            # You can adjust the sleep time as needed
            time.sleep(self.consolidation_interval)
            if not self.stop_consolidation_thread:
                self.consolidate_memories()

    def adaptive_consolidation_interval(self):
        """
        Adjusts the consolidation interval based on factors like:
        - Memory usage
        - Performance metrics (e.g., loss, accuracy)
        - Other relevant factors (e.g., stage of training)
        """

        # Example: Adjust based on memory pool usage and performance
        memory_stats = self.memory_pool.get_stats()
        current_usage = memory_stats['current_size'] / memory_stats['pool_size']
        
        # Example performance metric (e.g., get training loss from a global variable or callback)
        # Replace this with your actual performance tracking mechanism
        try:
          current_performance = self.get_latest_training_loss() # Assume you have a method to track this
        except:
          current_performance = 0.1 # Provide a default or fallback value

        # If memory usage is high and performance is poor, reduce the interval (more frequent consolidation)
        if current_usage > 0.8 and current_performance > 0.2: # Example thresholds
            self.consolidation_interval = max(
                self.min_consolidation_interval,
                int(self.consolidation_interval * (1 - self.consolidation_adaptation_rate))
            )
        # If memory usage is low and performance is good, increase the interval (less frequent consolidation)
        elif current_usage < 0.5 and current_performance < 0.1: # Example thresholds
            self.consolidation_interval = min(
                self.max_consolidation_interval,
                int(self.consolidation_interval * (1 + self.consolidation_adaptation_rate))
            )
        
        # Add other factors and adjustments as needed

        print(f"Consolidation interval updated to: {self.consolidation_interval}")

    def transfer_from_episodic_memory(self):
      """
      Transfers relevant experiences from episodic memory to the memory pool based on
      similarity to current memory pool states.
      """
      if not self.episodic_memory:
          return

      num_to_transfer = int(len(self.episodic_memory) * self.episodic_transfer_rate)
      if num_to_transfer == 0:
        return

      # Get a representative sample of embeddings from the memory pool
      memory_pool_samples = self.memory_pool.get_states()

      # Get a set of candidate experiences from episodic memory for transfer consideration
      candidate_experiences = self.episodic_memory.get_k_nearest_memories(
          query_embedding=memory_pool_samples.mean(dim=0),  # Use mean of memory pool samples as query
          k=num_to_transfer
      )

      experiences_to_transfer = []
      for experience in candidate_experiences:
          # Compute similarity between the experience embedding and memory pool samples
          experience_embedding = torch.cat([experience['goal'], experience['context']], dim=-1)

          similarity_scores = F.cosine_similarity(
              experience_embedding.unsqueeze(0),
              memory_pool_samples,
              dim=1
          )
          max_similarity = similarity_scores.max().item()

          # Transfer if similarity is above threshold
          if max_similarity > self.transfer_similarity_threshold:
              experiences_to_transfer.append(experience)

      # Update the memory pool with transferred experiences
      for experience in experiences_to_transfer:
          self.update_memory_pool_from_episodic(experience)

    def update_memory_pool_from_episodic(self, experience: Dict):
        """
        Updates the memory pool with an experience from episodic memory.

        Args:
            experience: A dictionary representing the experience from episodic memory.
                        Must have keys 'goal' and 'context' with corresponding tensor values.
        """

        # Convert experience to a suitable format for the memory pool
        # (This is highly dependent on your specific use case)
        # For example, you might use the concatenated goal and context as the state:
        episodic_state = torch.cat([experience['goal'], experience['context']], dim=-1)

        # 1. Find a similar state in the memory pool (if any)
        memory_states = self.memory_pool.get_states()
        similarity_scores = F.cosine_similarity(episodic_state.unsqueeze(0), memory_states, dim=1)
        most_similar_index = similarity_scores.argmax().item()
        max_similarity = similarity_scores[most_similar_index].item()

        # 2. Decide whether to replace the similar state or add a new one
        if max_similarity > self.transfer_similarity_threshold:
            # Replace the similar state if the new state has higher importance (or other criteria)
            # Example: Update importance based on episodic memory (you'll need a mechanism to assign importance in episodic memory)
            episodic_importance = experience.get('importance', 1.0)  # Get importance from episodic memory or default to 1.0

            current_importance = self.memory_pool.state_importance[most_similar_index].item()

            # Update importance with a weighted average (or other combination)
            updated_importance = (1 - self.importance_update_factor) * current_importance + self.importance_update_factor * episodic_importance

            if updated_importance > current_importance:
              self.memory_pool.memory_states[most_similar_index] = episodic_state
              self.memory_pool.state_importance[most_similar_index] = updated_importance
              # Update other relevant attributes (e.g., age, access count) if needed
              self.memory_pool.memory_age[most_similar_index] = 0  # Reset age for transferred state
        else:
            # Add as a new state if below capacity, otherwise, it might replace a low-importance state based on memory_pool's logic
            self.memory_pool.update(new_states=episodic_state.unsqueeze(0), k=1)

    def consolidate_memories(self):
        """
        Performs offline memory consolidation.
        This can include:
        - Re-clustering experiences in episodic memory.
        - Transferring consolidated experiences from episodic memory to the memory pool.
        - Adaptively adjusting the consolidation interval.
        - Other optimization or compression operations on the memory.
        """
        print("Consolidating memories...")

        # Re-cluster experiences in episodic memory
        self.episodic_memory.perform_cluster_maintenance()

        # Transfer relevant experiences from episodic memory to memory pool
        self.transfer_from_episodic_memory()

        # Adaptively adjust the consolidation interval
        self.adaptive_consolidation_interval()

        print("Memory consolidation complete.")
    

        # Track reasoning steps for CoT reward
        reasoning_steps = []
        confidence_scores = []
        
        # Compute n-gram hash embeddings with goal-aware processing
        batch_size = x.size(0)
        seq_len = x.size(1)
        byte_embeds = self.local_encoder['embedding'](x)
        
        # Enhance embeddings with goal information if provided
        if goal_context is not None:
          byte_embeds = byte_embeds + goal_context.unsqueeze(1).expand(-1, seq_len, -1)
        else:
          # If goal_context is not provided, encode the selected goals
          goal_context = self._encode_goals(selected_goals)
          byte_embeds = byte_embeds + goal_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate system prompt embeddings to byte embeddings
        if 'system_prompt' in x:
            system_prompt_embeds = x['system_prompt']['embeds']
            byte_embeds = torch.cat([system_prompt_embeds, byte_embeds], dim=1)  # Concatenate along the sequence length dimension
            
            # Adjust sequence length to account for the system prompt
            seq_len += system_prompt_embeds.size(1)

        # Add n-gram hash embeddings with CoT tracking
        for n in range(3, 9):  # n-grams from 3 to 8 as in paper
            # Create n-grams
            ngrams = []
            for i in range(seq_len - n + 1):
                ngram = x[:, i:i+n]  # [batch_size, n]
                # Compute hash (using FNV-1a for better distribution)
                ngram_hash = torch.zeros(batch_size, dtype=torch.long, device=x.device)
                FNV_PRIME = 1677
                FNV_OFFSET = 2166136

                 for j in range(n):
                    ngram_hash = ((ngram_hash * FNV_PRIME) % self.config.hash_vocab_size) ^ ngram[:, j]
                ngrams.append(ngram_hash)
            
            if ngrams:  # Only if we have n-grams
                ngram_tensor = torch.stack(ngrams, dim=1)  # [batch_size, seq_len-n+1]
                ngram_embeds = self.hash_embeddings[f'ngram_{n}'](ngram_tensor)  # [batch_size, seq_len-n+1, d_model]
                
                # Track reasoning step
                reasoning_steps.append({
                    'type': f'ngram_{n}',
                    'embeddings': ngram_embeds.detach(),
                    'description': f'Computing {n}-gram hash embeddings'
                })
                
                # Add to corresponding positions in byte embeddings with positional weighting
                for i in range(seq_len - n + 1):
                    # Weight based on position (center positions get higher weight)
                    pos_weights = torch.linspace(0.5, 1.0, n, device=x.device)
                    pos_weights = pos_weights.view(1, -1, 1)  # [1, n, 1]
                    byte_embeds[:, i:i+n] += ngram_embeds[:, i].unsqueeze(1) * pos_weights / n
                    
                # Compute confidence scores for this n-gram level
                with torch.no_grad():
                    logits = self.local_decoder['output'](ngram_embeds)
                    confidence = self.cot_reward.compute_confidence_score(logits)
                    confidence_scores.append(confidence)
        
        # Normalize embeddings
        byte_embeds = byte_embeds / (len(self.hash_embeddings) + 1)  # +1 for original byte embeddings
        
        # Process fMRI data if available
        if brain_regions is not None:
            # Project fMRI data to embedding space for each region
            region_embeds = {}
            for region, activity in brain_regions.items():
                # Project activity to region space
                region_embed = self.region_embeddings[region](activity)
                
                # Apply region-specific attention
                region_attn, _ = self.region_attention[region](
                    byte_embeds,
                    region_embed.unsqueeze(0),
                    region_embed.unsqueeze(0)
                )
                
                # Gate with activity level
                gate = torch.sigmoid(activity.mean())
                region_embeds[region] = gate * region_attn + (1 - gate) * byte_embeds
            
            # Fuse region embeddings with anatomical constraints
            fused_embeds = self._anatomically_constrained_fusion(region_embeds)
            
            # Combine with byte embeddings
            for region, embed in fused_embeds.items():
                # Weight based on region's relevance to current text segment
                relevance = torch.cosine_similarity(byte_embeds, embed, dim=-1)
                relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                byte_embeds = byte_embeds + relevance.unsqueeze(-1) * embed
        
        # Pass sensory, tactile, and audio data to the BrainRegionMapper along with other data
        region_embeds = self.region_mapper.map_to_regions(
            {'text_embeds': byte_embeds},
            fmri_data=brain_regions,
            region_texts=None,
            region_images=None,
            sensory_data=sensory_data,
            tactile_data=tactile_data,
            audio_data=audio_data
        )

        # Combine region embeddings with byte embeddings
        combined_embeds = byte_embeds
        for region, embed in region_embeds.items():
            combined_embeds = torch.cat([combined_embeds, embed.unsqueeze(1)], dim=1)

        # Use the combined embeddings as input to the encoder
        hidden = combined_embeds
        
        encoder_states = []
        
        # Process through encoder layers with fMRI integration and CoT tracking
        for layer_idx, (layer, cross_attn) in enumerate(zip(
            self.local_encoder['transformer'].layers,
            self.local_encoder['cross_attention']
        )):
            # Transformer layer
            hidden = layer(hidden)
            encoder_states.append(hidden)
            
            # Track reasoning step
            reasoning_steps.append({
                'type': f'encoder_layer_{layer_idx}',
                'embeddings': hidden.detach(),
                'description': f'Processing through encoder layer {layer_idx}'
            })
            
            # Compute confidence score for this layer
            with torch.no_grad():
                logits = self.local_decoder['output'](hidden)
                confidence = self.cot_reward.compute_confidence_score(logits)
                confidence_scores.append(confidence)
            
            # Compute entropy for dynamic patching
            entropy = self.entropy_model(hidden).squeeze(-1)
            patch_boundaries = entropy > self.config.entropy_threshold
            
            # Map text features to brain regions with CoT tracking
            if brain_regions is not None:
                # Project text features to each brain region's space based on region's function
                region_projections = {}
                for region, activity in brain_regions.items():
                    # Track reasoning step for this region
                    reasoning_steps.append({
                        'type': f'brain_region_{region}',
                        'embeddings': hidden.detach(),
                        'description': f'Processing {region} brain region features'
                    })
                    
                    # Project text to region space based on region's function
                    if region == 'visual':
                        # Visual regions process low-level features (bytes, chars)
                        region_text = self.region_embeddings[region](hidden[:, :2])
                        reasoning_steps.append({
                            'type': 'visual_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing low-level visual features'
                        })
                    elif region in ['language', 'semantic']:
                        # Language regions process words and sentences
                        region_text = self.region_embeddings[region](hidden[:, 2:4])
                        reasoning_steps.append({
                            'type': 'language_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing language and semantic features'
                        })
                    elif region in ['memory', 'executive']:
                        # Memory/executive regions process higher-level context
                        region_text = self.region_embeddings[region](hidden[:, 4:])
                        reasoning_steps.append({
                            'type': 'memory_executive_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing memory and executive control features'
                        })
                    else:
                        # Other regions process full sequence
                        region_text = self.region_embeddings[region](hidden)
                        reasoning_steps.append({
                            'type': f'general_processing_{region}',
                            'embeddings': region_text.detach(),
                            'description': f'Processing general features for {region}'
                        })
                    
                    # Compute confidence score for this region
                    with torch.no_grad():
                        logits = self.local_decoder['output'](region_text)
                        confidence = self.cot_reward.compute_confidence_score(logits)
                        confidence_scores.append(confidence)
                    
                    # Get region-specific attention with activity gating
                    region_attn, _ = self.region_attention[region](
                        region_text,
                        activity.unsqueeze(0),
                        activity.unsqueeze(0)
                    )
                    
                    # Apply activity-based gating
                    gate = torch.sigmoid(activity.mean())
                    region_projections[region] = gate * region_attn + (1 - gate) * region_text
                
                # Fuse region projections with anatomical constraints
                region_embeds = self._anatomically_constrained_fusion(region_projections)
                
                # Integrate region embeddings back into hidden states
                for region, embed in region_embeds.items():
                    # Weight based on region's relevance and hierarchical level
                    if region == 'visual':
                        # Visual regions influence early layers more
                        relevance = torch.cosine_similarity(hidden[:, :2], embed, dim=-1)
                    elif region in ['language', 'semantic']:
                        # Language regions influence middle layers
                        relevance = torch.cosine_similarity(hidden[:, 2:4], embed, dim=-1)
                    elif region in ['memory', 'executive']:
                        # Memory/executive regions influence later layers
                        relevance = torch.cosine_similarity(hidden[:, 4:], embed, dim=-1)
                    else:
                        # Other regions influence all layers
                        relevance = torch.cosine_similarity(hidden, embed, dim=-1)
                    
                    # Apply temperature scaling and integrate
                    relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                    hidden = hidden + relevance.unsqueeze(-1) * embed
            
            # Create patches based on entropy and process through RNN
            patches = []
            start_idx = 0
            for i, is_boundary in enumerate(patch_boundaries):
                if is_boundary or i == len(patch_boundaries) - 1:
                    if i + 1 - start_idx >= self.config.min_patch_size:
                        patch = hidden[start_idx:i+1]
                        if len(patch) <= self.config.max_patch_size:
                            # Get patch embedding
                            patch_embed = patch.mean(dim=0)
                            
            # Process through RNN and enhanced binary memory
            # 1. RNN processing for state tracking
            rnn_out, rnn_states = self.latent_transformer(
                patch_embed.unsqueeze(0),  # Add batch dimension
                rnn_states
            )
            
            # 2. Extract binary latent states
            binary_latents = self.memory_pool.state_encoder(rnn_out)
            binary_states = (binary_latents > 0.5).bool()
            
            # 3. Update and retrieve from binary memory pool with binary states
            self.memory_pool.update(rnn_out, self.config.memory_topk, binary_states)
            memory_states = self.memory_pool.get_states()
            
            # 4. Track binary latent statistics
            with torch.no_grad():
                binary_sparsity = 1.0 - binary_states.float().mean().item()
                binary_entropy = -torch.mean(
                    binary_states.float() * torch.log2(binary_states.float() + 1e-10) +
                    (1 - binary_states.float()) * torch.log2(1 - binary_states.float() + 1e-10)
                ).item()
                
                self.memory_pool.usage_stats['binary_stats'].append({
                    'sparsity': binary_sparsity,
                    'entropy': binary_entropy,
                    'active_bits': binary_states.sum().item(),
                    'total_bits': binary_states.numel()
                })
            
            # 3. Combine RNN and memory states with importance-based weighting
            memory_importance = torch.sigmoid(self.compression_policy(memory_states))
            combined_out = (
                (1 - memory_importance) * rnn_out + 
                memory_importance * memory_states
            )
            
            # 4. Apply attention-based state selection with memory integration
            if rnn_states is not None:
                # Get hidden states from all layers
                h_states = torch.stack([h for h, _ in rnn_states])  # [num_layers, batch, hidden]
                
                # Get memory states
                memory_keys = self.memory.key_embed1.weight  # Get memory keys
                
                # Concatenate RNN and memory states
                combined_states = torch.cat([
                    h_states.transpose(0, 1),  # [batch, num_layers, hidden]
                    memory_keys.unsqueeze(0).expand(h_states.size(1), -1, -1)  # [batch, num_keys, hidden]
                ], dim=1)
                
                # Compute attention scores over combined states
                attn_scores, _ = self.state_attention(
                    patch_embed.unsqueeze(0),  # Query: current patch
                    combined_states,  # Keys: RNN + memory states
                    combined_states   # Values: RNN + memory states
                )
                
                # Select states based on attention
                selected_state = (attn_scores @ combined_states).squeeze(0)
                patches.append(selected_state)
            else:
                patches.append(combined_out.squeeze(0))
            
            # Update start index for next patch
            start_idx = i + 1
                            
            
            # Handle case where no patches were created
            if not patches:
                patches = [hidden.mean(dim=0)]
            patches = torch.stack(patches)
            
            # Cross attention between bytes and patches
            hidden, _ = cross_attn(hidden, patches, patches)
        
        # Global latent transformer processing
        latent_states = self.latent_transformer(patches)
        
        # Local decoder processing with brain region integration
        decoder_states = []
        hidden = byte_embeds
        
        # Process through decoder layers with hierarchical brain region integration
        for layer_idx, (layer, cross_attn) in enumerate(zip(self.local_decoder['transformer'].layers, self.local_decoder['cross_attention'])):
            # Cross attention with patches
            hidden, _ = cross_attn(hidden, latent_states, latent_states)
            
            # Integrate brain region information if available
            if brain_regions is not None:
                # Project text features to each brain region's space based on layer depth
                region_projections = {}
                for region, activity in brain_regions.items():
                    # Early layers focus on low-level features
                    if layer_idx < len(self.local_decoder['transformer'].layers) // 3:
                        if region in ['visual', 'sensory_temporal', 'sensory_parietal']:
                            # Project early layers to sensory regions
                            region_text = self.region_embeddings[region](hidden[:, :2])
                    # Middle layers focus on language and semantic processing
                    elif layer_idx < 2 * len(self.local_decoder['transformer'].layers) // 3:
                        if region in ['language', 'semantic']:
                            # Project middle layers to language regions
                            region_text = self.region_embeddings[region](hidden[:, 2:4])
                    # Late layers focus on high-level integration
                    else:
                        if region in ['memory', 'executive', 'integration']:
                            # Project late layers to higher-order regions
                            region_text = self.region_embeddings[region](hidden[:, 4:])
                    
                    # Get region-specific attention with activity gating
                    region_attn, _ = self.region_attention[region](
                        region_text,
                        activity.unsqueeze(0),
                        activity.unsqueeze(0)
                    )
                    
                    # Apply activity-based gating
                    gate = torch.sigmoid(activity.mean())
                    region_projections[region] = gate * region_attn + (1 - gate) * region_text
                
                # Fuse region projections with anatomical constraints
                region_embeds = self._anatomically_constrained_fusion(region_projections)
                
                # Integrate region embeddings back into hidden states
                for region, embed in region_embeds.items():
                    # Weight based on region's relevance and layer depth
                    if layer_idx < len(self.local_decoder['transformer'].layers) // 3:
                        # Early layers: stronger sensory influence
                        if region in ['visual', 'sensory_temporal', 'sensory_parietal']:
                            relevance = torch.cosine_similarity(hidden[:, :2], embed, dim=-1)
                    elif layer_idx < 2 * len(self.local_decoder['transformer'].layers) // 3:
                        # Middle layers: stronger language influence
                        if region in ['language', 'semantic']:
                            relevance = torch.cosine_similarity(hidden[:, 2:4], embed, dim=-1)
                    else:
                        # Late layers: stronger high-level influence
                        if region in ['memory', 'executive', 'integration']:
                            relevance = torch.cosine_similarity(hidden[:, 4:], embed, dim=-1)
                    
                    # Apply temperature scaling and integrate
                    relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                    hidden = hidden + relevance.unsqueeze(-1) * embed
            
            # Transformer layer
            hidden = layer(hidden)
            decoder_states.append(hidden)
        
        # Predict next byte
        logits = self.local_decoder['output'](hidden)
        
        # FLAME factuality-aware alignment
        if self.training:
            # Compute factuality score for self-alignment
            factuality_score = self._compute_factuality_score(hidden)
            
            # Apply factuality-aware loss weighting
            if self.is_fact_based:  # For fact-based instructions
                # Use own knowledge (from pre-training) for supervision
                supervision_weight = factuality_score
            else:  # For non-fact-based instructions
                # Use human demonstrations for supervision
                supervision_weight = 1.0
                
            # Scale logits based on factuality alignment
            logits = logits * supervision_weight.unsqueeze(-1)
            
            # Store factuality metrics for monitoring
            self.factuality_metrics = {
                'factuality_score': factuality_score.mean().item(),
                'supervision_weight': supervision_weight.mean().item()
            }
        
        return {
            'logits': logits,
            'patches': patches,
            'latent_states': latent_states,
            'encoder_states': encoder_states,
            'decoder_states': decoder_states,
            'entropy': entropy,
            'patch_boundaries': patch_boundaries,
            'factuality_metrics': getattr(self, 'factuality_metrics', None)
        }
        # Update combined input for the next subgoal, including the current output and memory states
        combined_input = current_output
        if hasattr(self, 'memory_pool'):
            memory_states = self.memory_pool.get_states()
            if memory_states.numel() > 0:
                combined_input = torch.cat([combined_input, memory_states], dim=0)

    # After executing all subgoals, compute the final CoT reward based on the entire reasoning path
    final_reward = self.cot_reward.compute_reward(torch.stack([step['output'] for step in reasoning_steps if 'output' in step]))

    return current_output, reasoning_steps, final_reward
        # N-gram hash embeddings
        self.hash_embeddings = nn.ModuleDict({
            f'ngram_{n}': nn.Embedding(config.hash_vocab_size, config.d_model)
            for n in range(3, 9)  # n-grams from 3 to 8 as in paper
        })
        
        # Local encoder for byte-level processing
        self.local_encoder = nn.ModuleDict({
            'embedding': nn.Embedding(256, config.d_model),
            'transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                num_layers=config.encoder_layers
            ),
            'cross_attention': nn.ModuleList([
                nn.MultiheadAttention(
                    config.d_model,
                    config.n_heads,
                    dropout=config.dropout,
                    batch_first=True
                ) for _ in range(config.encoder_layers)
            ])
        })
        
        # Global latent transformer combining MSRNN and shared memory
        self.latent_transformer = MultiStateRNN(
            hidden_size=config.d_model,
            num_layers=config.n_layers
        )
        
        # Shared memory for efficient compression and retrieval
        self.memory = ProductKeyMemory(
            dim=config.d_model,
            num_keys=config.memory_size,  # Default 1024 half-keys
            topk=config.memory_topk,      # Default 32
            add_silu=True
        )
        
        # Memory compression policy
        self.compression_policy = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )
        
        # Attention for state selection
        self.state_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # State compression policy
        self.compression_policy = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )
        
        # Local decoder for byte-level generation
        self.local_decoder = nn.ModuleDict({
            'transformer': nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                num_layers=config.decoder_layers
            ),
            'cross_attention': nn.ModuleList([
                nn.MultiheadAttention(
                    config.d_model,
                    config.n_heads,
                    dropout=config.dropout,
                    batch_first=True
                ) for _ in range(config.decoder_layers)
            ]),
            'output': nn.Linear(config.d_model, 256)  # Predict next byte
        })
        
        # Entropy model for dynamic patching
        self.entropy_model = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )
        
        # Region embeddings for brain integration
        self.region_embeddings = nn.ModuleDict({
            region: nn.Linear(config.d_model, config.region_dim)
            for region in [
                'visual', 'language', 'memory', 'attention', 'executive',
                'semantic', 'integration', 'sensory_temporal', 'sensory_parietal'
            ]
        })
        
        # Region-specific attention
        self.region_attention = nn.ModuleDict({
            region: nn.MultiheadAttention(
                config.region_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )
            for region in self.region_embeddings.keys()
        })
        
        # Cross-region fusion
        self.region_fusion = CrossModalFusion(
            config.region_dim,
            config.n_heads,
            config.dropout,
            len(self.region_embeddings)
        )
        
        # Anatomical distance matrix (from BrainLM paper coordinates)
        self.register_buffer('anatomical_distances', self._compute_anatomical_distances())
        
    def _compute_anatomical_distances(self) -> torch.Tensor:
        """Compute pairwise anatomical distances between brain regions"""
        regions = list(self.region_embeddings.keys())
        n_regions = len(regions)
        distances = torch.zeros(n_regions, n_regions)
        
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i != j:
                    # Get coordinates for both regions
                    coords1 = torch.tensor(self.regions[region1]['mni_coords'])
                    coords2 = torch.tensor(self.regions[region2]['mni_coords'])
                    
                    # Compute minimum distance between any pair of coordinates
                    pairwise_dist = torch.cdist(coords1, coords2)
                    min_dist = pairwise_dist.min().item()
                    
                    # Convert to weight (closer = higher weight)
                    distances[i, j] = 1.0 / (1.0 + min_dist / 50.0)  # 50mm normalization
        
        return distances
        
    def _anatomically_constrained_fusion(
        self,
        region_embeds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Fuse region embeddings while respecting anatomical constraints"""
        regions = list(region_embeds.keys())
        n_regions = len(regions)
        
        # Stack embeddings
        stacked_embeds = torch.stack([region_embeds[r] for r in regions])  # [n_regions, seq_len, d_model]
        
        # Get anatomical weights for these regions
        region_indices = [list(self.region_embeddings.keys()).index(r) for r in regions]
        weights = self.anatomical_distances[region_indices][:, region_indices]  # [n_regions, n_regions]
        
        # Apply pathway-specific weights from BrainLM paper
        pathway_weights = {
            ('visual', 'semantic'): 1.2,      # Strong visual-semantic pathway
            ('semantic', 'language'): 1.2,    # Strong semantic-language pathway
            ('language', 'memory'): 1.1,      # Language-memory integration
            ('memory', 'executive'): 1.1,     # Memory-executive control
            ('attention', 'executive'): 1.2,  # Strong attention-executive pathway
            ('integration', 'semantic'): 1.1, # Integration with semantics
            ('integration', 'memory'): 1.1,   # Integration with memory
            ('integration', 'executive'): 1.1, # Integration with control
            ('sensory_temporal', 'language'): 1.2,  # Strong connection for sensory processing
            ('sensory_temporal', 'semantic'): 1.2,  # Sensory-semantic integration
            ('sensory_temporal', 'memory'): 1.1,    # Sensory memory integration
            ('sensory_parietal', 'attention'): 1.2, # Strong connection for tactile attention
            ('sensory_parietal', 'executive'): 1.1  # Executive control of tactile processing
        }
        
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if (region1, region2) in pathway_weights:
                    weights[i, j] *= pathway_weights[(region1, region2)]
                elif (region2, region1) in pathway_weights:
                    weights[i, j] *= pathway_weights[(region2, region1)]
        
        # Additional boost for integration region
        integration_idx = [i for i, r in enumerate(regions) if r == 'integration']
        if integration_idx:
            weights[integration_idx, :] *= 1.1  # 10% boost for integration pathways
            weights[:, integration_idx] *= 1.1
        
        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted fusion
        fused_embeds = {}
        for i, region in enumerate(regions):
            # Compute weighted sum of embeddings
            weighted_sum = (weights[i].unsqueeze(-1).unsqueeze(-1) * stacked_embeds).sum(dim=0)
            
            # Add residual connection
            fused_embeds[region] = weighted_sum + region_embeds[region]
        
        return fused_embeds

    def forward(self, x: torch.Tensor, goal_context: Optional[torch.Tensor] = None, brain_regions: Optional[Dict[str, torch.Tensor]] = None, rnn_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass implementing BLT workflow with RNN processing, brain region integration, and SELFGOAL"""
        # Initialize SELFGOAL for this forward pass if not already initialized
        if not hasattr(self, 'current_goal'):
            # Extract high-level goal from input context
            self.current_goal = self._extract_high_level_goal(x)
            self.goal_manager.initialize_tree(self.current_goal)

        # Pass sensory, tactile, and audio data to the BrainRegionMapper along with other data
        region_embeds = self.region_mapper.map_to_regions(
            token_embeds,
            fmri_data=brain_regions,  # Assuming this is how you pass fMRI data
            region_texts=None,  # You might need to adjust this
            region_images=None,  # You might need to adjust this
            sensory_data=sensory_data,
            tactile_data=tactile_data,
            audio_data=audio_data
        )
            
        # Get current context for SELFGOAL
        context = {
            'input': x,
            'brain_regions': brain_regions,
            'rnn_states': rnn_states,
            'step': getattr(self, 'step_counter', 0)
        }
        
        # Select relevant subgoals using SELFGOAL
        #selected_goals = self.goal_manager.select_goals(context) - Removed because this is already present in the Multi-RNN network COCONUT trainer thinking proceses. 
        
        # Track reasoning steps for CoT reward
        reasoning_steps = []
        confidence_scores = []
        
        # Compute n-gram hash embeddings with goal-aware processing
        batch_size = x.size(0)
        seq_len = x.size(1)
        byte_embeds = self.local_encoder['embedding'](x)
        
        # Enhance embeddings with goal information if provided
        if goal_context is not None:
          byte_embeds = byte_embeds + goal_context.unsqueeze(1).expand(-1, seq_len, -1)
        else:
          # If goal_context is not provided, encode the selected goals
          goal_context = self._encode_goals(selected_goals)
          byte_embeds = byte_embeds + goal_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate system prompt embeddings to byte embeddings
        if 'system_prompt' in x:
            system_prompt_embeds = x['system_prompt']['embeds']
            byte_embeds = torch.cat([system_prompt_embeds, byte_embeds], dim=1)  # Concatenate along the sequence length dimension
            
            # Adjust sequence length to account for the system prompt
            seq_len += system_prompt_embeds.size(1)

        # Add n-gram hash embeddings with CoT tracking
        for n in range(3, 9):  # n-grams from 3 to 8 as in paper
            # Create n-grams
            ngrams = []
            for i in range(seq_len - n + 1):
                ngram = x[:, i:i+n]  # [batch_size, n]
                # Compute hash (using FNV-1a for better distribution)
                ngram_hash = torch.zeros(batch_size, dtype=torch.long, device=x.device)
                FNV_PRIME = 16777619
                FNV_OFFSET = 2166136261
                for j in range(n):
                    ngram_hash = ((ngram_hash * FNV_PRIME) % self.config.hash_vocab_size) ^ ngram[:, j]
                ngrams.append(ngram_hash)
            
            if ngrams:  # Only if we have n-grams
                ngram_tensor = torch.stack(ngrams, dim=1)  # [batch_size, seq_len-n+1]
                ngram_embeds = self.hash_embeddings[f'ngram_{n}'](ngram_tensor)  # [batch_size, seq_len-n+1, d_model]
                
                # Track reasoning step
                reasoning_steps.append({
                    'type': f'ngram_{n}',
                    'embeddings': ngram_embeds.detach(),
                    'description': f'Computing {n}-gram hash embeddings'
                })
                
                # Add to corresponding positions in byte embeddings with positional weighting
                for i in range(seq_len - n + 1):
                    # Weight based on position (center positions get higher weight)
                    pos_weights = torch.linspace(0.5, 1.0, n, device=x.device)
                    pos_weights = pos_weights.view(1, -1, 1)  # [1, n, 1]
                    byte_embeds[:, i:i+n] += ngram_embeds[:, i].unsqueeze(1) * pos_weights / n
                    
                # Compute confidence scores for this n-gram level
                with torch.no_grad():
                    logits = self.local_decoder['output'](ngram_embeds)
                    confidence = self.cot_reward.compute_confidence_score(logits)
                    confidence_scores.append(confidence)
        
        # Normalize embeddings
        byte_embeds = byte_embeds / (len(self.hash_embeddings) + 1)  # +1 for original byte embeddings
        
        # Process fMRI data if available
        if brain_regions is not None:
            # Project fMRI data to embedding space for each region
            region_embeds = {}
            for region, activity in brain_regions.items():
                # Project activity to region space
                region_embed = self.region_embeddings[region](activity)
                
                # Apply region-specific attention
                region_attn, _ = self.region_attention[region](
                    byte_embeds,
                    region_embed.unsqueeze(0),
                    region_embed.unsqueeze(0)
                )
                
                # Gate with activity level
                gate = torch.sigmoid(activity.mean())
                region_embeds[region] = gate * region_attn + (1 - gate) * byte_embeds
            
            # Fuse region embeddings with anatomical constraints
            fused_embeds = self._anatomically_constrained_fusion(region_embeds)
            
            # Combine with byte embeddings
            for region, embed in fused_embeds.items():
                # Weight based on region's relevance to current text segment
                relevance = torch.cosine_similarity(byte_embeds, embed, dim=-1)
                relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                byte_embeds = byte_embeds + relevance.unsqueeze(-1) * embed
        
        encoder_states = []
        hidden = byte_embeds
        
        # Process through encoder layers with fMRI integration and CoT tracking
        for layer_idx, (layer, cross_attn) in enumerate(zip(
            self.local_encoder['transformer'].layers,
            self.local_encoder['cross_attention']
        )):
            # Transformer layer
            hidden = layer(hidden)
            encoder_states.append(hidden)
            
            # Track reasoning step
            reasoning_steps.append({
                'type': f'encoder_layer_{layer_idx}',
                'embeddings': hidden.detach(),
                'description': f'Processing through encoder layer {layer_idx}'
            })
            
            # Compute confidence score for this layer
            with torch.no_grad():
                logits = self.local_decoder['output'](hidden)
                confidence = self.cot_reward.compute_confidence_score(logits)
                confidence_scores.append(confidence)
            
            # Compute entropy for dynamic patching
            entropy = self.entropy_model(hidden).squeeze(-1)
            patch_boundaries = entropy > self.config.entropy_threshold
            
            # Map text features to brain regions with CoT tracking
            if brain_regions is not None:
                # Project text features to each brain region's space based on region's function
                region_projections = {}
                for region, activity in brain_regions.items():
                    # Track reasoning step for this region
                    reasoning_steps.append({
                        'type': f'brain_region_{region}',
                        'embeddings': hidden.detach(),
                        'description': f'Processing {region} brain region features'
                    })
                    
                    # Project text to region space based on region's function
                    if region == 'visual':
                        # Visual regions process low-level features (bytes, chars)
                        region_text = self.region_embeddings[region](hidden[:, :2])
                        reasoning_steps.append({
                            'type': 'visual_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing low-level visual features'
                        })
                    elif region in ['language', 'semantic']:
                        # Language regions process words and sentences
                        region_text = self.region_embeddings[region](hidden[:, 2:4])
                        reasoning_steps.append({
                            'type': 'language_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing language and semantic features'
                        })
                    elif region in ['memory', 'executive']:
                        # Memory/executive regions process higher-level context
                        region_text = self.region_embeddings[region](hidden[:, 4:])
                        reasoning_steps.append({
                            'type': 'memory_executive_processing',
                            'embeddings': region_text.detach(),
                            'description': 'Processing memory and executive control features'
                        })
                    else:
                        # Other regions process full sequence
                        region_text = self.region_embeddings[region](hidden)
                        reasoning_steps.append({
                            'type': f'general_processing_{region}',
                            'embeddings': region_text.detach(),
                            'description': f'Processing general features for {region}'
                        })
                    
                    # Compute confidence score for this region
                    with torch.no_grad():
                        logits = self.local_decoder['output'](region_text)
                        confidence = self.cot_reward.compute_confidence_score(logits)
                        confidence_scores.append(confidence)
                    
                    # Get region-specific attention with activity gating
                    region_attn, _ = self.region_attention[region](
                        region_text,
                        activity.unsqueeze(0),
                        activity.unsqueeze(0)
                    )
                    
                    # Apply activity-based gating
                    gate = torch.sigmoid(activity.mean())
                    region_projections[region] = gate * region_attn + (1 - gate) * region_text
                
                # Fuse region projections with anatomical constraints
                region_embeds = self._anatomically_constrained_fusion(region_projections)
                
                # Integrate region embeddings back into hidden states
                for region, embed in region_embeds.items():
                    # Weight based on region's relevance and hierarchical level
                    if region == 'visual':
                        # Visual regions influence early layers more
                        relevance = torch.cosine_similarity(hidden[:, :2], embed, dim=-1)
                    elif region in ['language', 'semantic']:
                        # Language regions influence middle layers
                        relevance = torch.cosine_similarity(hidden[:, 2:4], embed, dim=-1)
                    elif region in ['memory', 'executive']:
                        # Memory/executive regions influence later layers
                        relevance = torch.cosine_similarity(hidden[:, 4:], embed, dim=-1)
                    else:
                        # Other regions influence all layers
                        relevance = torch.cosine_similarity(hidden, embed, dim=-1)
                    
                    # Apply temperature scaling and integrate
                    relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                    hidden = hidden + relevance.unsqueeze(-1) * embed
            
            # Create patches based on entropy and process through RNN
            patches = []
            start_idx = 0
            for i, is_boundary in enumerate(patch_boundaries):
                if is_boundary or i == len(patch_boundaries) - 1:
                    if i + 1 - start_idx >= self.config.min_patch_size:
                        patch = hidden[start_idx:i+1]
                        if len(patch) <= self.config.max_patch_size:
                            # Get patch embedding
                            patch_embed = patch.mean(dim=0)
                            
            # Process through RNN and enhanced binary memory
            # 1. RNN processing for state tracking
            rnn_out, rnn_states = self.latent_transformer(
                patch_embed.unsqueeze(0),  # Add batch dimension
                rnn_states
            )
            
            # 2. Extract binary latent states
            binary_latents = self.memory_pool.state_encoder(rnn_out)
            binary_states = (binary_latents > 0.5).bool()
            
            # 3. Update and retrieve from binary memory pool with binary states
            self.memory_pool.update(rnn_out, self.config.memory_topk, binary_states)
            memory_states = self.memory_pool.get_states()
            
            # 4. Track binary latent statistics
            with torch.no_grad():
                binary_sparsity = 1.0 - binary_states.float().mean().item()
                binary_entropy = -torch.mean(
                    binary_states.float() * torch.log2(binary_states.float() + 1e-10) +
                    (1 - binary_states.float()) * torch.log2(1 - binary_states.float() + 1e-10)
                ).item()
                
                self.memory_pool.usage_stats['binary_stats'].append({
                    'sparsity': binary_sparsity,
                    'entropy': binary_entropy,
                    'active_bits': binary_states.sum().item(),
                    'total_bits': binary_states.numel()
                })
            
            # 3. Combine RNN and memory states with importance-based weighting
            memory_importance = torch.sigmoid(self.compression_policy(memory_states))
            combined_out = (
                (1 - memory_importance) * rnn_out + 
                memory_importance * memory_states
            )
            
            # 4. Apply attention-based state selection with memory integration
            if rnn_states is not None:
                # Get hidden states from all layers
                h_states = torch.stack([h for h, _ in rnn_states])  # [num_layers, batch, hidden]
                
                # Get memory states
                memory_keys = self.memory.key_embed1.weight  # Get memory keys
                
                # Concatenate RNN and memory states
                combined_states = torch.cat([
                    h_states.transpose(0, 1),  # [batch, num_layers, hidden]
                    memory_keys.unsqueeze(0).expand(h_states.size(1), -1, -1)  # [batch, num_keys, hidden]
                ], dim=1)
                
                # Compute attention scores over combined states
                attn_scores, _ = self.state_attention(
                    patch_embed.unsqueeze(0),  # Query: current patch
                    combined_states,  # Keys: RNN + memory states
                    combined_states   # Values: RNN + memory states
                )
                
                # Select states based on attention
                selected_state = (attn_scores @ combined_states).squeeze(0)
                patches.append(selected_state)
            else:
                patches.append(combined_out.squeeze(0))
            
            # Update start index for next patch
            start_idx = i + 1
                            
            
            # Handle case where no patches were created
            if not patches:
                patches = [hidden.mean(dim=0)]
            patches = torch.stack(patches)
            
            # Cross attention between bytes and patches
            hidden, _ = cross_attn(hidden, patches, patches)
        
        # Global latent transformer processing
        latent_states = self.latent_transformer(patches)
        
        # Local decoder processing with brain region integration
        decoder_states = []
        hidden = byte_embeds
        
        # Process through decoder layers with hierarchical brain region integration
        for layer_idx, (layer, cross_attn) in enumerate(zip(self.local_decoder['transformer'].layers, self.local_decoder['cross_attention'])):
            # Cross attention with patches
            hidden, _ = cross_attn(hidden, latent_states, latent_states)
            
            # Integrate brain region information if available
            if brain_regions is not None:
                # Project text features to each brain region's space based on layer depth
                region_projections = {}
                for region, activity in brain_regions.items():
                    # Early layers focus on low-level features
                    if layer_idx < len(self.local_decoder['transformer'].layers) // 3:
                        if region in ['visual', 'sensory_temporal', 'sensory_parietal']:
                            # Project early layers to sensory regions
                            region_text = self.region_embeddings[region](hidden[:, :2])
                    # Middle layers focus on language and semantic processing
                    elif layer_idx < 2 * len(self.local_decoder['transformer'].layers) // 3:
                        if region in ['language', 'semantic']:
                            # Project middle layers to language regions
                            region_text = self.region_embeddings[region](hidden[:, 2:4])
                    # Late layers focus on high-level integration
                    else:
                        if region in ['memory', 'executive', 'integration']:
                            # Project late layers to higher-order regions
                            region_text = self.region_embeddings[region](hidden[:, 4:])
                    
                    # Get region-specific attention with activity gating
                    region_attn, _ = self.region_attention[region](
                        region_text,
                        activity.unsqueeze(0),
                        activity.unsqueeze(0)
                    )
                    
                    # Apply activity-based gating
                    gate = torch.sigmoid(activity.mean())
                    region_projections[region] = gate * region_attn + (1 - gate) * region_textinfo['levels']]).mean(0)
            
            # Project text to region space
            region_text_embed = self.region_embeddings[region](region_text)
            
            if fmri_data is not None:
                # Extract fMRI activity for this region's coordinates
                region_activity = self._extract_region_activity(
                    fmri_data, 
                    info['mni_coords'],
                    info['activity_threshold']
                )
                
                # Project fMRI activity to embedding space
                region_fmri_embed = self.fmri_projections[region](region_activity)
                
                # Attention-guided fusion of text and fMRI
                region_embed, _ = self.region_attention[region](
                    region_text_embed.unsqueeze(0),
                    region_fmri_embed.unsqueeze(0),
                    region_fmri_embed.unsqueeze(0)
                )
                region_embed = region_embed.squeeze(0)
                
                # Apply activity-based gating
                gate = torch.sigmoid(region_activity.mean())
                region_embed = gate * region_embed + (1 - gate) * region_text_embed
            else:
                region_embed = region_text_embed
            
            region_embeds[region] = region_embed
        
        # Cross-region fusion with anatomical constraints
        region_embeds = self._anatomically_constrained_fusion(region_embeds)
        
        return region_embeds
        
    def _extract_region_activity(
        self,
        fmri_data: torch.Tensor,
        coords: List[Tuple[int, int, int]],
        threshold: float
    ) -> torch.Tensor:
        """Extract and normalize fMRI activity for given coordinates"""
        # Extract activity from coordinates
        region_activity = []
        for x, y, z in coords:
            # Get activity at coordinate
            activity = fmri_data[..., x, y, z]
            region_activity.append(activity)
        
        # Stack and normalize
        region_activity = torch.stack(region_activity, dim=-1)
        region_activity = torch.nn.functional.normalize(region_activity, dim=-1)
        
        # Apply threshold
        region_activity = region_activity * (region_activity > threshold).float()
        
        return region_activity
        
    def _anatomically_constrained_fusion(
        self,
        region_embeds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Fuse region embeddings while respecting anatomical constraints"""
        # Define anatomical constraints from BrainLM paper
        adjacency = {
            'visual': ['semantic', 'integration'],  # Visual to semantic pathway
            'language': ['semantic', 'memory', 'integration', 'sensory_temporal'],  # Language processing pathway
            'memory': ['language', 'executive', 'integration', 'sensory_temporal'],  # Memory consolidation
            'attention': ['executive', 'integration', 'sensory_parietal'],  # Attention control
            'executive': ['memory', 'attention', 'integration'],  # Executive control
            'semantic': ['visual', 'language', 'integration', 'sensory_temporal'],  # Semantic processing
            'integration': ['visual', 'language', 'memory', 'attention', 'executive', 'semantic', 'sensory_temporal', 'sensory_parietal'],  # Integration hub
            'sensory_temporal': ['language', 'semantic', 'memory', 'integration'],  # Temporal lobe sensory processing
            'sensory_parietal': ['attention', 'integration']  # Parietal lobe sensory processing
        }
        
        # Constrained fusion
        fused_embeds = {}
        for region in region_embeds:
            # Get adjacent regions
            adjacent = adjacency[region]
            
            # Weighted sum with adjacent regions
            region_embed = region_embeds[region]
            for adj_region in adjacent:
                if adj_region in region_embeds:
                    # Weight based on anatomical distance
                    weight = self._compute_anatomical_weight(region, adj_region)
                    region_embed = region_embed + weight * region_embeds[adj_region]
            
            fused_embeds[region] = region_embed
            
        return fused_embeds
        
    def _compute_anatomical_weight(
        self,
        region1: str,
        region2: str
    ) -> float:
        """Compute weight based on anatomical distance and functional connectivity from BrainLM paper"""
        # Get all coordinates for both regions
        coords1 = self.regions[region1]['mni_coords']
        coords2 = self.regions[region2]['mni_coords']
        
        # Compute minimum distance between any pair of coordinates
        min_distance = float('inf')
        for c1 in coords1:
            for c2 in coords2:
                distance = sum((x1 - x2) ** 2 for x1, x2 in zip(c1, c2)) ** 0.5
                min_distance = min(min_distance, distance)
        
        # Base weight on distance (closer = higher weight)
        distance_weight = 1.0 / (1.0 + min_distance / 50.0)  # 50mm normalization factor
        
        # Additional weighting based on functional pathways from BrainLM paper
        pathway_weights = {
            ('visual', 'semantic'): 1.2,      # Strong visual-semantic pathway
            ('semantic', 'language'): 1.2,    # Strong semantic-language pathway
            ('language', 'memory'): 1.1,      # Language-memory integration
            ('memory', 'executive'): 1.1,     # Memory-executive control
            ('attention', 'executive'): 1.2,  # Strong attention-executive pathway
            ('integration', 'semantic'): 1.1, # Integration with semantics
            ('integration', 'memory'): 1.1,   # Integration with memory
            ('integration', 'executive'): 1.1, # Integration with control
            
            # Sensory temporal pathways
            ('sensory_temporal', 'language'): 1.2,  # Strong connection for taste/smell/sound processing
            ('sensory_temporal', 'semantic'): 1.2,  # Strong semantic integration of sensory information
            ('sensory_temporal', 'memory'): 1.1,    # Memory integration of sensory experiences
            ('integration', 'sensory_temporal'): 1.1, # Integration of temporal sensory information
            
            # Sensory parietal pathways  
            ('sensory_parietal', 'attention'): 1.2,  # Strong connection for tactile attention
            ('sensory_parietal', 'executive'): 1.1,  # Executive control of tactile processing
            ('integration', 'sensory_parietal'): 1.1  # Integration of parietal sensory information
        }
        
        # Get pathway weight if it exists (in either direction)
        pathway_weight = pathway_weights.get(
            (region1, region2),
            pathway_weights.get((region2, region1), 1.0)
        )
        
        # Combine distance and pathway weights
        weight = distance_weight * pathway_weight
        
        # Additional boost for integration region connections
        if 'integration' in (region1, region2):
            weight *= 1.1  # 10% boost for integration pathways
            
        return weight

class MultimodalBrainAwareDataset(Dataset):
    """Dataset for multimodal brain data with hierarchical structure and fMRI integration"""
    def __init__(
        self,
        text_data,
        fmri_data,
        config: TrainingConfig,
        augment_prob: float = 0.5,  # Probability of applying augmentation
        mask_prob: float = 0.15,  # Probability of masking tokens
        fmri_window: int = 200,  # Time window for fMRI samples (from paper)
        fmri_patch_size: int = 20,  # Patch size for temporal signals (from paper)
        n_parcels: int = 424  # Number of parcels for fMRI compression (from paper)
    ):
        self.text_data = text_data
        self.fmri_data = fmri_data
        self.config = config
        self.augment_prob = augment_prob
        self.mask_prob = mask_prob
        self.fmri_window = fmri_window
        self.fmri_patch_size = fmri_patch_size
        self.n_parcels = n_parcels
        self.system_prompt = system_prompt
        
        # fMRI processing components
        self.fmri_compressor = nn.Linear(fmri_data.shape[-1], n_parcels) if fmri_data is not None else None
        self.temporal_embeddings = nn.Parameter(torch.randn(fmri_window, config.d_model))
        self.spatial_embeddings = nn.Parameter(torch.randn(n_parcels, config.d_model))
        self.patch_projection = nn.Linear(fmri_patch_size, config.d_model)
        
        # Initialize factuality components
        self.factuality_reward = FactualityRewardModel(config)
        self.is_fact_based = False  # Will be set during training
        
        # Initialize fMRI encoder/decoder
        if fmri_data is not None:
            self.fmri_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                num_layers=config.encoder_layers
            )
            
            self.fmri_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                num_layers=config.decoder_layers
            )
        
        # Special tokens
        self.mask_token = config.mask_token
        self.pad_token = config.pad_token
        
        # Hierarchical masking strategies
        self.masking_strategies = {
            'bytes': self._mask_bytes,
            'chars': self._mask_chars,
            'words': self._mask_words,
            'sentences': self._mask_sentences,
            'paragraphs': self._mask_paragraphs
        }
        
        # Level-specific masking probabilities
        self.level_mask_probs = {
            'bytes': 0.15,      # More frequent masking at lower levels
            'chars': 0.15,
            'words': 0.12,
            'sentences': 0.10,
            'paragraphs': 0.08  # Less frequent masking at higher levels
        }
        
        # Cross-level masking probabilities
        self.cross_level_mask_prob = 0.05  # Probability of masking corresponding tokens
        
        # Use entropy-based tokenizer for hierarchical processing
        self.tokenizer = EntropyBasedTokenizer(config)
        
        # Brain region mapper
        self.region_mapper = BrainRegionMapper(config)
        
        
        # Hierarchical augmentation methods
        self.byte_augmentations = [
            self._byte_substitution,
            self._byte_insertion,
            self._byte_deletion
        ]
        
        self.char_augmentations = [
            self._char_substitution,
            self._char_swapping,
            self._char_deletion
        ]
        
        self.word_augmentations = [
            self._word_substitution,
            self._word_insertion,
            self._word_deletion,
            self._word_shuffling
        ]
        
        self.sentence_augmentations = [
            self._sentence_paraphrase,
            self._sentence_splitting,
            self._sentence_combination
        ]
        
        self.paragraph_augmentations = [
            self._paragraph_restructuring,
            self._paragraph_summarization,
            self._paragraph_expansion
        ]

    def _apply_hierarchical_augmentation(
        self,
        text: str,
        fmri: Optional[torch.Tensor] = None
    ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply hierarchical data augmentation"""
        if torch.rand(1).item() > self.augment_prob:
            return text, fmri
            
        # Text augmentation
        # 1. Byte-level
        if torch.rand(1).item() < 0.2:  # 20% chance
            aug_fn = torch.randint(0, len(self.byte_augmentations), (1,)).item()
            text = self.byte_augmentations[aug_fn](text)
        
        # 2. Character-level
        if torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.char_augmentations), (1,)).item()
            text = self.char_augmentations[aug_fn](text)
        
        # 3. Word-level
        if torch.rand(1).item() < 0.4:  # 40% chance
            aug_fn = torch.randint(0, len(self.word_augmentations), (1,)).item()
            text = self.word_augmentations[aug_fn](text)
        
        # 4. Sentence-level
        if torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.sentence_augmentations), (1,)).item()
            text = self.sentence_augmentations[aug_fn](text)
        
        # 5. Paragraph-level
        if torch.rand(1).item() < 0.2:  # 20% chance
            aug_fn = torch.randint(0, len(self.paragraph_augmentations), (1,)).item()
            text = self.paragraph_augmentations[aug_fn](text)
        
        return text, fmri
    
    # Byte-level augmentations
    def _byte_substitution(self, text: str) -> str:
        """Randomly substitute bytes"""
        bytes_list = list(text.encode())
        for i in range(len(bytes_list)):
            if torch.rand(1).item() < 0.1:  # 10% chance per byte
                bytes_list[i] = torch.randint(0, 256, (1,)).item()
        return bytes(bytes_list).decode(errors='ignore')
    
    def _byte_insertion(self, text: str) -> str:
        """Insert random bytes"""
        bytes_list = list(text.encode())
        for i in range(len(bytes_list)):
            if torch.rand(1).item() < 0.05:  # 5% chance per position
                bytes_list.insert(i, torch.randint(0, 256, (1,)).item())
        return bytes(bytes_list).decode(errors='ignore')
    
    def _byte_deletion(self, text: str) -> str:
        """Delete random bytes"""
        bytes_list = list(text.encode())
        return bytes([b for b in bytes_list if torch.rand(1).item() > 0.05]).decode(errors='ignore')
    
    # Character-level augmentations
    def _char_substitution(self, text: str) -> str:
        """Substitute characters with similar ones"""
        char_map = {
            'a': 'áàâä', 'e': 'éèêë', 'i': 'íìîï',
            'o': 'óòôö', 'u': 'úùûü', 'n': 'ñ'
        }
        result = ''
        for c in text:
            if c.lower() in char_map and torch.rand(1).item() < 0.1:
                options = char_map[c.lower()]
                result += options[torch.randint(0, len(options), (1,)).item()]
            else:
                result += c
        return result
    
    def _char_swapping(self, text: str) -> str:
        """Swap adjacent characters"""
        chars = list(text)
        for i in range(len(chars)-1):
            if torch.rand(1).item() < 0.1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
        return ''.join(chars)
    
    def _char_deletion(self, text: str) -> str:
        """Delete random characters"""
        return ''.join(c for c in text if torch.rand(1).item() > 0.05)
    
    # Word-level augmentations
    def _word_substitution(self, text: str) -> str:
        """Substitute words with synonyms"""
        # Simple synonym map (expand this with a proper synonym database)
        synonyms = {
            'happy': ['joyful', 'glad', 'pleased'],
            'sad': ['unhappy', 'depressed', 'down'],
            'big': ['large', 'huge', 'enormous']
        }
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms and torch.rand(1).item() < 0.2:
                options = synonyms[word.lower()]
                words[i] = options[torch.randint(0, len(options), (1,)).item()]
        return ' '.join(words)
    
    def _word_insertion(self, text: str) -> str:
        """Insert common words"""
        common_words = ['the', 'a', 'an', 'and', 'or', 'but']
        words = text.split()
        for i in range(len(words)):
            if torch.rand(1).item() < 0.1:
                words.insert(i, common_words[torch.randint(0, len(common_words), (1,)).item()])
        return ' '.join(words)
    
    def _word_deletion(self, text: str) -> str:
        """Delete random words"""
        words = text.split()
        return ' '.join(w for w in words if torch.rand(1).item() > 0.1)
    
    def _word_shuffling(self, text: str) -> str:
        """Shuffle words while preserving some local order"""
        words = text.split()
        for i in range(0, len(words)-2, 2):
            if torch.rand(1).item() < 0.2:
                words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)
    
    # Sentence-level augmentations
    def _sentence_paraphrase(self, text: str) -> str:
        """Simple rule-based paraphrasing"""
        # Add basic paraphrasing rules (expand this)
        patterns = [
            ('I am', "I'm"),
            ('They are', "They're"),
            ('will not', "won't"),
            ('cannot', "can't")
        ]
        for original, replacement in patterns:
            if torch.rand(1).item() < 0.3:
                text = text.replace(original, replacement)
        return text
    
    def _sentence_splitting(self, text: str) -> str:
        """Split long sentences"""
        sentences = text.split('. ')
        result = []
        for sent in sentences:
            if len(sent.split()) > 10 and torch.rand(1).item() < 0.3:
                words = sent.split()
                mid = len(words) // 2
                result.append(' '.join(words[:mid]) + '.')
                result.append(' '.join(words[mid:]) + '.')
            else:
                result.append(sent + '.')
        return ' '.join(result)
    
    def _sentence_combination(self, text: str) -> str:
        """Combine short sentences"""
        sentences = text.split('. ')
        result = []
        i = 0
        while i < len(sentences):
            if i < len(sentences)-1 and len(sentences[i].split()) < 5 and len(sentences[i+1].split()) < 5 and torch.rand(1).item() < 0.3:
                result.append(sentences[i] + ' and ' + sentences[i+1] + '.')
                i += 2
            else:
                result.append(sentences[i] + '.')
                i += 1
        return ' '.join(result)
    
    # Paragraph-level augmentations
    def _paragraph_restructuring(self, text: str) -> str:
        """Restructure paragraphs"""
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # Randomly swap paragraphs
            for i in range(len(paragraphs)-1):
                if torch.rand(1).item() < 0.2:
                    paragraphs[i], paragraphs[i+1] = paragraphs[i+1], paragraphs[i]
        return '\n\n'.join(paragraphs)
    
    def _paragraph_summarization(self, text: str) -> str:
        """Simple extractive summarization"""
        paragraphs = text.split('\n\n')
        result = []
        for para in paragraphs:
            sentences = para.split('. ')
            if len(sentences) > 3 and torch.rand(1).item() < 0.2:
                # Keep first and last sentences, and a random middle one
                middle_idx = torch.randint(1, len(sentences)-1, (1,)).item()
                selected = [sentences[0], sentences[middle_idx], sentences[-1]]
                result.append('. '.join(selected) + '.')
            else:
                result.append(para)
        return '\n\n'.join(result)
    
    def _paragraph_expansion(self, text: str) -> str:
        """Expand paragraphs with common phrases"""
        common_phrases = [
            'In other words,',
            'For example,',
            'Additionally,',
            'Furthermore,',
            'Moreover,'
        ]
        paragraphs = text.split('\n\n')
        result = []
        for para in paragraphs:
            if torch.rand(1).item() < 0.2:
                phrase = common_phrases[torch.randint(0, len(common_phrases), (1,)).item()]
                sentences = para.split('. ')
                insert_idx = torch.randint(0, len(sentences), (1,)).item()
                sentences.insert(insert_idx, phrase)
                result.append('. '.join(sentences))
            else:
                result.append(para)
        return '\n\n'.join(result)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Get raw data
        text = self.text_data[idx]
        task = row['task'] #Added for SDV Dataset Labeling for self-configuration at runtime. 
        input_text = row['input']
        target_text = row['target']

        inputs = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        targets = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        # Add system prompt to tokens and token_embeds
        tokens['system_prompt'] = self.tokenizer.tokenize(self.system_prompt)
        token_embeds['system_prompt'] = self.tokenizer.embed(tokens['system_prompt'])

            return {
            "task": task,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "target": targets.input_ids.squeeze(0) # Or any other representation of the target
        }

        
        # Process fMRI data if available
        if self.fmri_data is not None:
            # Get fMRI data and compress to parcels
            fmri = self.fmri_data[idx]
            fmri = self.fmri_compressor(fmri)  # [time, n_parcels]
            
            # Randomly trim to window size
            if fmri.size(0) > self.fmri_window:
                start_idx = torch.randint(0, fmri.size(0) - self.fmri_window, (1,))
                fmri = fmri[start_idx:start_idx + self.fmri_window]
            
            # Split temporal signals into patches
            n_patches = self.fmri_window // self.fmri_patch_size
            fmri_patches = fmri.unfold(0, self.fmri_patch_size, self.fmri_patch_size)
            fmri_patches = fmri_patches.transpose(1, 2)  # [n_patches, patch_size, n_parcels]
            
            # Project patches to token space
            fmri_tokens = self.patch_projection(fmri_patches)  # [n_patches, n_parcels, d_model]
            
            # Add spatial and temporal embeddings
            positions = torch.arange(n_patches).unsqueeze(-1) * self.fmri_patch_size
            fmri_tokens = fmri_tokens + self.temporal_embeddings[positions]
            fmri_tokens = fmri_tokens + self.spatial_embeddings.unsqueeze(0)
            
            # Apply masking if needed
            if torch.rand(1).item() < self.mask_prob:
                mask = torch.rand(fmri_tokens.size(0)) < self.mask_prob
                fmri_tokens_masked = fmri_tokens.clone()
                fmri_tokens_masked[mask] = 0
                
                # Encode visible tokens
                encoded = self.fmri_encoder(fmri_tokens_masked)
                
                # Decode full sequence
                decoded = self.fmri_decoder(fmri_tokens, encoded)
                # Compute reconstruction loss only for masked tokens
                recon_loss = nn.MSELoss()(decoded[mask], fmri_tokens[mask])
            else:
                fmri_tokens_masked = fmri_tokens
                encoded = self.fmri_encoder(fmri_tokens)
                decoded = self.fmri_decoder(fmri_tokens, encoded)
                recon_loss = nn.MSELoss()(decoded, fmri_tokens)
        else:
            fmri = None
            fmri_tokens = None
            fmri_tokens_masked = None
            encoded = None
            decoded = None
            recon_loss = 0.0
        
        # Hierarchical tokenization
        tokens = self.tokenizer.tokenize(text)
        token_embeds = self.tokenizer.embed(tokens)
        
        # Map to brain regions using processed fMRI
        region_embeds = self.region_mapper.map_to_regions(
            token_embeds, fmri_tokens if fmri is not None else None
        )
        
        return {
            'tokens': tokens,
            'token_embeds': token_embeds,
            'region_embeds': region_embeds,
            'fmri': fmri,
            'fmri_tokens': fmri_tokens,
            'fmri_tokens_masked': fmri_tokens_masked,
            'fmri_encoded': encoded,
            'fmri_decoded': decoded,
            'fmri_recon_loss': recon_loss,
            'system_prompt': system_prompt
        }

class TrainingConfig:
    """Configuration for training"""
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        # rStar-Math parameters
        r_star_rounds: int = 4,                # Number of self-evolution rounds
        initial_mcts_rollouts: int = 16,      # Initial number of MCTS rollouts per problem
        increased_mcts_rollouts: int = 64,   # Increased number of MCTS rollouts for challenging problems
        mcts_expansion_factor: int = 8,          # Number of candidate nodes to explore at each MCTS step
        mcts_c: float = 2.0,                  # Exploration constant in the UCT formula
        code_cot_verification: bool = True,   # Flag to enable code-augmented CoT and verification

       # Episodic Memory Parameters
        episodic_memory_capacity: int = 1000,
        memory_cluster_method: str = 'hdbscan',
        hdbscan_min_cluster_size: int = 5,
        hdbscan_metric: str = 'euclidean',
        hdbscan_cluster_selection_epsilon: float = 0.0,
        som_grid_size: int = 10,
        som_input_dim: int = 1024,  # Adjust based on your embedding size
        som_sigma: float = 1.0,
        som_learning_rate: float = 0.5,
        cluster_relevance_threshold: float = 0.7,
        cluster_centroid_update_freq: int = 10,
        max_cluster_size: int = 50,
        merge_similarity_threshold: float = 0.9,
        enable_cluster_maintenance: bool = True,
        cluster_maintenance_interval: int = 100,
        hierarchical_clustering_method: str = 'hdbscan',
        memory_importance_decay: float = 0.99,
        expansion_threshold: float = 0.2,
        num_retrieve_memories: int = 5,
        expansion_check_interval: int = 100,
        max_neurons: int = 10000,

        # Goal Manager parameters
        max_goals: int = 5,
        min_importance: float = 0.1,
        exploration_factor: float = 0.5,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.99,
        similarity_threshold: float = 0.85,

        # rStar-Math parameters
        self.r_star_rounds = r_star_rounds
        self.initial_mcts_rollouts = initial_mcts_rollouts
        self.increased_mcts_rollouts = increased_mcts_rollouts
        self.mcts_expansion_factor = mcts_expansion_factor
        self.mcts_c = mcts_c
        self.code_cot_verification = code_cot_verification
        
        # Goal Manager parameters
        self.max_goals = max_goals
        self.min_importance = min_importance
        self.exploration_factor = exploration_factor
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.similarity_threshold = similarity_threshold

        # Episodic memory parameters
        self.episodic_memory_capacity = episodic_memory_capacity
        self.memory_cluster_method = memory_cluster_method
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon
        self.som_grid_size = som_grid_size
        self.som_input_dim = som_input_dim
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.cluster_relevance_threshold = cluster_relevance_threshold
        self.cluster_centroid_update_freq = cluster_centroid_update_freq
        self.max_cluster_size = max_cluster_size
        self.merge_similarity_threshold = merge_similarity_threshold
        self.enable_cluster_maintenance = enable_cluster_maintenance
        self.cluster_maintenance_interval = cluster_maintenance_interval
        self.hierarchical_clustering_method = hierarchical_clustering_method
        self.memory_importance_decay = memory_importance_decay
        self.expansion_threshold = expansion_threshold
        self.num_retrieve_memories = num_retrieve_memories
        self.expansion_check_interval = expansion_check_interval
        self.max_neurons = max_neurons

        distillation_config: Optional[Dict[str, Any]] = {
            'enabled': True,
            'temperature': 2.0,        # Temperature for softening distributions
            'alpha': 0.5,              # Weight for distillation loss
            'hierarchical_temp': {     # Level-specific temperatures
                'bytes': 1.0,          # Sharper for low-level features
                'chars': 1.5,
                'words': 2.0,
                'sentences': 2.5,
                'paragraphs': 3.0      # Softer for high-level features
            },
            'feature_matching': {      # Feature-level distillation
                'enabled': True,
                'layers': ['hidden', 'attention', 'ffn'],
                'weight': 0.1
            },
            'attention_transfer': {    # Attention-based transfer
                'enabled': True,
                'weight': 0.1,
                'types': ['self', 'cross']
            },
            'progressive_stages': {    # Progressive knowledge transfer
                'freeze_lower': True,  # Freeze lower levels while training higher
                'unfreeze_ratio': 0.2  # Ratio of training to start unfreezing
            },
            'level_specific': {        # Level-specific distillation settings
                'bytes_to_chars': {
                    'weight': 1.0,
                    'features': ['hidden', 'attention'],
                    'temp': 1.5
                },
                'chars_to_words': {
                    'weight': 0.8,
                    'features': ['hidden', 'attention'],
                    'temp': 2.0
                },
                'words_to_sentences': {
                    'weight': 0.6,
                    'features': ['hidden', 'attention'],
                    'temp': 2.5
                },
                'sentences_to_paragraphs': {
                    'weight': 0.4,
                    'features': ['hidden', 'attention'],
                    'temp': 3.0
                }
            }
        },
        curriculum_config: Optional[Dict[str, Any]] = {
            'schedule': {
                'bytes': 0.0,      # Start with bytes
                'chars': 0.2,      # Add chars at 20% of training
                'words': 0.4,      # Add words at 40% of training
                'sentences': 0.6,  # Add sentences at 60% of training
                'paragraphs': 0.8  # Add paragraphs at 80% of training
            },
            'difficulty_metrics': {
                'bytes': ['entropy', 'length'],
                'chars': ['vocab_size', 'complexity'],
                'words': ['frequency', 'length'],
                'sentences': ['depth', 'branching'],
                'paragraphs': ['coherence', 'structure']
            },
            'progression_criteria': {
                'performance_threshold': 0.8,  # Accuracy threshold to advance
                'stability_window': 100,      # Steps to confirm stability
                'min_samples': 1000,          # Min samples before advancing
                'max_difficulty': 0.9         # Max difficulty to attempt
            },
            'pacing_strategy': {
                'type': 'adaptive',           # adaptive or fixed
                'warmup_steps': 1000,         # Steps before adaptation
                'growth_rate': 0.1,           # Difficulty growth rate
                'decay_rate': 0.05,           # Difficulty decay rate
                'update_freq': 100            # Steps between updates
            },
            'level_prerequisites': {
                'chars': ['bytes'],
                'words': ['chars'],
                'sentences': ['words'],
                'paragraphs': ['sentences']
            },
            'sampling_weights': {
                'performance': 0.4,    # Weight based on accuracy
                'complexity': 0.3,     # Weight based on difficulty
                'novelty': 0.3         # Weight based on uniqueness
            },
            'curriculum_metrics': {
                'track_confusion': True,     # Track confusion matrix
                'monitor_gradients': True,   # Monitor gradient flow
                'analyze_errors': True,      # Analyze error patterns
                'measure_retention': True    # Track knowledge retention
            }
        },
        gradient_config: Optional[Dict[str, Any]] = {
            'accumulation_steps': {
                'bytes': 1,        # No accumulation needed
                'chars': 2,        # Accumulate over 2 steps
                'words': 4,        # Accumulate over 4 steps
                'sentences': 8,    # Accumulate over 8 steps
                'paragraphs': 16   # Accumulate over 16 steps
            },
            'batch_multipliers': {
                'bytes': 1.0,      # Base batch size
                'chars': 0.8,      # 80% of base batch size
                'words': 0.6,      # 60% of base batch size
                'sentences': 0.4,  # 40% of base batch size
                'paragraphs': 0.2  # 20% of base batch size
            },
            'gradient_scaling': {
                'enabled': True,
                'base_scale': 1.0,
                'level_scales': {
                    'bytes': 1.0,      # Base gradient scale
                    'chars': 1.2,      # Scale up for fewer samples
                    'words': 1.5,      # Scale up more
                    'sentences': 2.0,  # Scale up further
                    'paragraphs': 2.5  # Scale up most
                }
            },
            'sync_gradients': True,  # Synchronize gradients across levels
            'normalize_grads': True,  # Normalize gradients by sequence length
            'clip_mode': 'global'     # Global gradient clipping
        },
        loss_config: Optional[Dict[str, Any]] = {
            'level_weights': {
                'bytes': 1.0,      # Base level weight
                'chars': 1.2,      # Slightly higher for character understanding
                'words': 1.5,      # Important for semantic meaning
                'sentences': 1.8,  # Critical for context
                'paragraphs': 2.0  # Highest for global understanding
            },
            'dynamic_scaling': {
                'enabled': True,
                'window_size': 100,  # Steps for moving average
                'scale_factor': 0.1  # Scale factor for dynamic weights
            },
            'gradient_balancing': {
                'enabled': True,
                'norm_type': 2,    # L2 normalization
                'clip_value': 1.0  # Maximum gradient norm
            },
            'level_lr_multipliers': {
                'bytes': 1.0,      # Base learning rate
                'chars': 0.9,      # Slightly lower for stability
                'words': 0.8,      # Lower for word-level
                'sentences': 0.7,  # Lower for sentence-level
                '
                # Fuse region projections with anatomical constraints
                region_embeds = self._anatomically_constrained_fusion(region_projections)
                
                # Integrate region embeddings back into hidden states
                for region, embed in region_embeds.items():
                    # Weight based on region's relevance and layer depth
                    if layer_idx < len(self.local_decoder['transformer'].layers) // 3:
                        # Early layers: stronger sensory influence
                        if region in ['visual', 'sensory_temporal', 'sensory_parietal']:
                            relevance = torch.cosine_similarity(hidden[:, :2], embed, dim=-1)
                    elif layer_idx < 2 * len(self.local_decoder['transformer'].layers) // 3:
                        # Middle layers: stronger language influence
                        if region in ['language', 'semantic']:
                            relevance = torch.cosine_similarity(hidden[:, 2:4], embed, dim=-1)
                    else:
                        # Late layers: stronger high-level influence
                        if region in ['memory', 'executive', 'integration']:
                            relevance = torch.cosine_similarity(hidden[:, 4:], embed, dim=-1)
                    
                    # Apply temperature scaling and integrate
                    relevance = torch.softmax(relevance / 0.1, dim=0)  # Temperature of 0.1
                    hidden = hidden + relevance.unsqueeze(-1) * embed
            
            # Transformer layer
            hidden = layer(hidden)
            decoder_states.append(hidden)
        
        # Predict next byte
        logits = self.local_decoder['output'](hidden)
        
        # FLAME factuality-aware alignment
        if self.training:
            # Compute factuality score for self-alignment
            factuality_score = self._compute_factuality_score(hidden)
            
            # Apply factuality-aware loss weighting
            if self.is_fact_based:  # For fact-based instructions
                # Use own knowledge (from pre-training) for supervision
                supervision_weight = factuality_score
            else:  # For non-fact-based instructions
                # Use human demonstrations for supervision
                supervision_weight = 1.0
                
            # Scale logits based on factuality alignment
            logits = logits * supervision_weight.unsqueeze(-1)
            
            # Store factuality metrics for monitoring
            self.factuality_metrics = {
                'factuality_score': factuality_score.mean().item(),
                'supervision_weight': supervision_weight.mean().item()
            }
        
        return {
            'logits': logits,
            'patches': patches,
            'latent_states': latent_states,
            'encoder_states': encoder_states,
            'decoder_states': decoder_states,
            'entropy': entropy,
            'patch_boundaries': patch_boundaries,
            'factuality_metrics': getattr(self, 'factuality_metrics', None)
        }
        
    def _compute_factuality_score(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute factuality score for self-alignment using FLAME approach"""
        # Project hidden states to factuality space
        factuality_hidden = self.factuality_projection(hidden)
        
        # Compute attention-based factuality scores
        scores, _ = self.factuality_attention(
            factuality_hidden, factuality_hidden, factuality_hidden
        )
        
        # Aggregate scores across attention heads
        scores = scores.mean(dim=1)  # Average across heads
        
        # Apply temperature scaling
        scores = torch.sigmoid(scores / self.factuality_temperature)
        
        return scores

class EntropyBasedTokenizer(HierarchicalTokenizer):
    """Extends hierarchical tokenizer with entropy-based patching"""
    def __init__(self, config):
        super().__init__(config)
        
        # Entropy model for dynamic patching
        self.entropy_model = EntropyModel(config)
        
        # Entropy thresholds
        self.global_threshold = config.entropy_threshold
        self.relative_threshold = config.relative_threshold
        
        # Context window for entropy estimation
        self.context_window = config.window_size
        
    def compute_patch_boundaries(self, bytes_tensor: torch.Tensor) -> torch.Tensor:
        """Find patch boundaries based on entropy"""
        # Get entropy estimates
        with torch.no_grad():
            entropies = self.entropy_model(bytes_tensor)
        
        # Find boundaries using global threshold
        global_boundaries = entropies > self.global_threshold
        
        # Find boundaries using relative threshold
        entropy_diff = entropies[1:] - entropies[:-1]
        relative_boundaries = torch.zeros_like(entropies, dtype=torch.bool)
        relative_boundaries[1:] = entropy_diff > self.relative_threshold
        
        # Combine boundary methods
        boundaries = global_boundaries | relative_boundaries
        
        return boundaries
        
    def create_patches(self, bytes_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Create patches based on entropy boundaries"""
        boundaries = self.compute_patch_boundaries(bytes_tensor)
        
        # Split into patches
        patches = []
        start_idx = 0
        
        for i, is_boundary in enumerate(boundaries):
            if is_boundary or i == len(boundaries) - 1:
                # Enforce min/max patch sizes
                if i + 1 - start_idx >= self.config.min_patch_size:
                    patch = bytes_tensor[start_idx:i+1]
                    if len(patch) <= self.config.max_patch_size:
                        patches.append(patch)
                        start_idx = i + 1
                
        return patches

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Override tokenize to use entropy-based patching"""
        # Get base hierarchical tokenization
        tokens = super().tokenize(text)
        
        # Add entropy-based patches
        bytes_tensor = tokens['bytes']
        patches = self.create_patches(bytes_tensor)
        
        # Convert patches to tensor
        max_patch_size = max(len(p) for p in patches)
        padded_patches = torch.zeros(len(patches), max_patch_size, dtype=torch.long)
        for i, patch in enumerate(patches):
            padded_patches[i, :len(patch)] = patch
            
        tokens['entropy_patches'] = padded_patches
        
        return tokens

class HierarchicalTokenizer:
    """Hierarchical tokenizer that builds from bytes to higher-level structures"""
    def __init__(self, config):
        self.config = config
        self.byte_vocab_size = 256
        self.char_vocab_size = 128  # ASCII
        self.word_vocab_size = config.hash_vocab_size
        self.sentence_vocab_size = config.hash_vocab_size
        self.paragraph_vocab_size = config.hash_vocab_size
        
        # Byte to char mappings
        self.byte_to_char = {i: chr(i) for i in range(128)}
        
        # Hash functions for higher levels
        self.word_hash = lambda x: hash(x) % self.word_vocab_size
        self.sent_hash = lambda x: hash(x) % self.sentence_vocab_size
        self.para_hash = lambda x: hash(x) % self.paragraph_vocab_size
        
        # Embeddings for each level
        self.byte_embeddings = nn.Embedding(self.byte_vocab_size, config.d_model)
        self.char_embeddings = nn.Embedding(self.char_vocab_size, config.d_model) 
        self.word_embeddings = nn.Embedding(self.word_vocab_size, config.d_model)
        self.sent_embeddings = nn.Embedding(self.sentence_vocab_size, config.d_model)
        self.para_embeddings = nn.Embedding(self.paragraph_vocab_size, config.d_model)
        
        # Hierarchical positional encodings
        self.level_embeddings = nn.Embedding(5, config.d_model)  # 5 levels
        self.relative_pos_embeddings = nn.ModuleDict({
            level: nn.Embedding(2 * config.window_size - 1, config.d_model)
            for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        })
        self.hierarchical_pos_mlp = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text into hierarchical levels"""
        # Convert to bytes
        bytes_tensor = torch.tensor([ord(c) for c in text])
        
        # Group into chars (ASCII)
        chars = []
        for b in bytes_tensor:
            if b < 128:  # ASCII range
                chars.append(self.byte_to_char[b.item()])
        chars_tensor = torch.tensor([ord(c) for c in chars])
        
        # Group into words
        words = ''.join([self.byte_to_char[b.item()] for b in bytes_tensor]).split()
        words_tensor = torch.tensor([self.word_hash(w) for w in words])
        
        # Group into sentences (split on ., !, ?)
        sents = []
        curr_sent = []
        for w in words:
            curr_sent.append(w)
            if w[-1] in '.!?':
                sents.append(' '.join(curr_sent))
                curr_sent = []
        if curr_sent:
            sents.append(' '.join(curr_sent))
        sents_tensor = torch.tensor([self.sent_hash(s) for s in sents])
        
        # Group into paragraphs (split on newlines)
        paras = ' '.join(sents).split('\n')
        paras_tensor = torch.tensor([self.para_hash(p) for p in paras])
        
        return {
            'bytes': bytes_tensor,
            'chars': chars_tensor, 
            'words': words_tensor,
            'sentences': sents_tensor,
            'paragraphs': paras_tensor
        }

    def embed(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get embeddings for each hierarchical level"""
        return {
            'bytes': self.byte_embeddings(tokens['bytes']),
            'chars': self.char_embeddings(tokens['chars']),
            'words': self.word_embeddings(tokens['words']), 
            'sentences': self.sent_embeddings(tokens['sentences']),
            'paragraphs': self.para_embeddings(tokens['paragraphs'])
        }

class BrainRegionMapper:
    """Maps brain regions to hierarchical text representations based on fMRI activity"""
    def __init__(self, config):
        self.config = config
        
        # Define brain regions and their functions from BrainLM paper
        self.regions = {
            'visual': {
                'levels': ['bytes', 'chars'],  # Low-level visual features
                'mni_coords': [(25, -80, 5), (-25, -80, 5)],  # V1/V2 coordinates
                'activity_threshold': 0.6,
                'functions': ['pattern recognition', 'feature detection']
            },
            'language': {
                'levels': ['words', 'sentences'],  # Language processing
                'mni_coords': [
                    (45, 15, 15),   # Broca's area (left)
                    (-45, 15, 15),  # Wernicke's area (left)
                    (50, -55, 25),  # Angular gyrus (left)
                    (-50, -55, 25)  # Angular gyrus (right)
                ],
                'activity_threshold': 0.7,
                'functions': ['semantic processing', 'syntax', 'comprehension']
            },
            'memory': {
                'levels': ['paragraphs'],  # Long-term memory and context
                'mni_coords': [
                    (30, -20, -10),  # Hippocampus (left)
                    (-30, -20, -10), # Hippocampus (right)
                    (35, -40, -15),  # Parahippocampal (left)
                    (-35, -40, -15)  # Parahippocampal (right)
                ],
                'activity_threshold': 0.65,
                'functions': ['episodic memory', 'contextual binding']
            },
            'attention': {
                'levels': ['sentences', 'paragraphs'],  # Attention and control
                'mni_coords': [
                    (35, 45, 30),   # DLPFC (left)
                    (-35, 45, 30),  # DLPFC (right)
                    (40, 30, 45),   # Frontal eye fields
                    (-40, 30, 45)   # Frontal eye fields
                ],
                'activity_threshold': 0.75,
                'functions': ['selective attention', 'cognitive control']
            },
            'executive': {
                'levels': ['paragraphs'],  # High-level planning
                'mni_coords': [
                    (40, 35, 30),   # Anterior PFC (left)
                    (-40, 35, 30),  # Anterior PFC (right)
                    (45, 40, 35),   # Frontopolar cortex
                    (-45, 40, 35)   # Frontopolar cortex
                ],
                'activity_threshold': 0.8,
                'functions': ['planning', 'reasoning', 'abstraction']
            },
            'semantic': {
                'levels': ['words', 'sentences'],  # Meaning and concepts
                'mni_coords': [
                    (55, -45, -15),  # Temporal pole (left)
                    (-55, -45, -15), # Temporal pole (right)
                    (50, -35, -5),   # Inferior temporal
                    (-50, -35, -5)   # Inferior temporal
                ],
                'activity_threshold': 0.7,
                'functions': ['semantic memory', 'concept representation']
            },
            'integration': {
                'levels': ['sentences', 'paragraphs'],  # Information integration
                'mni_coords': [
                    (45, -60, 35),  # Angular gyrus (left)
                    (-45, -60, 35), # Angular gyrus (right)
                    (40, -65, 40),  # Posterior parietal
                    (-40, -65, 40)  # Posterior parietal
                ],
                'activity_threshold': 0.75,
                'functions': ['multimodal integration', 'abstraction']
            },
            'sensory_temporal': {
                'levels': ['words', 'sentences'],  # Sensory processing in temporal lobe
                'mni_coords': [
                    (40, -15, -20),  # Superior temporal gyrus (left)
                    (-40, -15, -20), # Superior temporal gyrus (right)
                    (45, -25, -15),  # Middle temporal gyrus (left)
                    (-45, -25, -15)  # Middle temporal gyrus (right)
                ],
                'activity_threshold': 0.65,
                'functions': ['taste processing', 'smell processing', 'auditory integration']
            },
            'sensory_parietal': {
                'levels': ['words', 'sentences'],  # Sensory processing in parietal lobe
                'mni_coords': [
                    (35, -30, 50),   # Primary somatosensory cortex (left)
                    (-35, -30, 50),  # Primary somatosensory cortex (right)
                    (40, -35, 55),   # Secondary somatosensory cortex (left)
                    (-40, -35, 55)   # Secondary somatosensory cortex (right)
                ],
                'activity_threshold': 0.65,
                'functions': ['tactile processing', 'proprioception', 'sensory integration']
            }
        }
        
        # Embeddings and projections for each region
        self.region_embeddings = nn.ModuleDict({
            region: nn.Linear(config.d_model, config.region_dim)
            for region in self.regions.keys()
        })
        
        # fMRI activity projections
        self.fmri_projections = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(len(coords), config.region_dim),
                nn.LayerNorm(config.region_dim),
                nn.ReLU(),
                nn.Linear(config.region_dim, config.region_dim)
            )
            for region, info in self.regions.items()
            for coords in [info['mni_coords']]
        })
        
        # Region-specific attention for activity-guided mapping
        self.region_attention = nn.ModuleDict({
            region: nn.MultiheadAttention(
                config.region_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )
            for region in self.regions.keys()
        })
        
        # Cross-attention for region fusion
        self.region_fusion = CrossModalFusion(
            config.region_dim,
            config.n_heads,
            config.dropout,
            len(self.regions)
        )

    def map_to_regions(self,
                      text_embeds: Dict[str, torch.Tensor],
                      fmri_data: Optional[torch.Tensor] = None
                      ) -> Dict[str, torch.Tensor]:
        """Map hierarchical text embeddings to brain regions guided by fMRI activity"""
        region_embeds = {}

def map_to_regions(self,
                      text_embeds: Dict[str, torch.Tensor],
                      fmri_data: Optional[Dict[str, torch.Tensor]] = None,
                      region_texts: Optional[Dict[str, str]] = None,
                      region_images: Optional[Dict[str, torch.Tensor]] = None,
                      sensory_data: Optional[torch.Tensor] = None,
                      tactile_data: Optional[torch.Tensor] = None,
                      audio_data: Optional[torch.Tensor] = None
                      ) -> Dict[str, torch.Tensor]:
        """
        Map hierarchical text embeddings, fMRI data, region texts, images, sensory data, tactile data, and audio data to brain regions.
        """
        region_embeds = {}

        for region, info in self.regions.items():
            # Initialize a list to store all relevant embeddings for the current region
            combined_embeddings = []

            # Always include text embeddings relevant to the region
            region_text_embeds = torch.stack([
                text_embeds[level].mean(0) for level in info['levels']
            ]).mean(0)
            combined_embeddings.append(region_text_embeds)

            # Add fMRI data processing if available and relevant to the region
            if fmri_data is not None and region in fmri_data:
                region_fmri_data = fmri_data[region]
                region_fmri_embed = self.fmri_projections[region](region_fmri_data)
                combined_embeddings.append(region_fmri_embed)

            # Add region-specific text data processing if available
            if region_texts is not None and region in region_texts:
                region_text = region_texts[region]
                region_text_tokens = self.config.tokenizer.tokenize(region_text)
                region_text_embed = self.config.tokenizer.embed(region_text_tokens)
                region_text_embed = region_text_embed['words'].mean(dim=0)  # Example: Mean of word embeddings
                combined_embeddings.append(region_text_embed)

            # Add region-specific image data processing if available
            if region_images is not None and region in region_images:
                region_image = region_images[region]
                region_image_embed = self.image_encoder(region_image.unsqueeze(0)).squeeze(0)  # Process image
                combined_embeddings.append(region_image_embed)

            # Add sensory data processing for the sensory_parietal region
            if sensory_data is not None and region == 'sensory_parietal':
                sensory_embed = self.region_embeddings[region](sensory_data)
                combined_embeddings.append(sensory_embed)

            # Add tactile data processing for the sensory_temporal region
            if tactile_data is not None and region == 'sensory_temporal':
                tactile_embed = self.region_embeddings[region](tactile_data)
                combined_embeddings.append(tactile_embed)

            # Add audio data processing if available and relevant to the region
            if audio_data is not None and region in ['language', 'executive', 'sensory_temporal']:
                audio_embed = self.audio_encoder(audio_data.unsqueeze(0)).squeeze(0)  # Process audio
                combined_embeddings.append(audio_embed)

            # Concatenate all collected embeddings for this region
            combined_embedding = torch.cat(combined_embeddings, dim=-1)

            # Project combined embedding to region space
            region_embed = self.region_embeddings[region](combined_embedding)

            # Apply region-specific attention (if needed) #This needs some fleshed out code. 
            # ... (You might want to add an attention mechanism here)

            region_embeds[region] = region_embed

        # Cross-region fusion with anatomical constraints
        region_embeds = self._anatomically_constrained_fusion(region_embeds)

        return region_embeds

 def map_to_regions(self,
                      text_embeds: Dict[str, torch.Tensor],
                      fmri_data: Optional[torch.Tensor] = None,
                      region_texts: Optional[Dict[str, str]] = None,
                      region_images: Optional[Dict[str, torch.Tensor]] = None
                      ) -> Dict[str, torch.Tensor]:
        """Map hierarchical text embeddings, fMRI data, region texts, and images to brain regions."""
        region_embeds = {}

        #Allow the fmri scans to take image and text embeddings from labeled datasets for training. 
        for region, info in self.regions.items():
            # Get text embeddings for this region's levels
            region_text_embeds = torch.stack([
                text_embeds[level].mean(0) for level in info['levels']
            ]).mean(0)

            # Concatenate text, fMRI, and image embeddings (if available)
            combined_embedding = region_text_embeds

            if fmri_data is not None and region in fmri_data:
                region_fmri_data = fmri_data[region]
                region_fmri_embed = self.fmri_projections[region](region_fmri_data)
                combined_embedding = torch.cat([combined_embedding, region_fmri_embed], dim=-1)

            if region_texts is not None and region in region_texts:
                region_text = region_texts[region]
                region_text_tokens = self.config.tokenizer.tokenize(region_text)
                region_text_embed = self.config.tokenizer.embed(region_text_tokens)
                # Assuming you want to use a mean of word embeddings for the text
                region_text_embed = region_text_embed['words'].mean(dim=0)
                combined_embedding = torch.cat([combined_embedding, region_text_embed], dim=-1)

            if region_images is not None and region in region_images:
                region_image = region_images[region]
                region_image_embed = self.image_encoder(region_image.unsqueeze(0)).squeeze(0)  # Add and remove batch dimension
                combined_embedding = torch.cat([combined_embedding, region_image_embed], dim=-1)

            # Project combined embedding to region space
            region_embed = self.region_embeddings[region](combined_embedding)

            # Apply region-specific attention (if needed)
            # ...

            region_embeds[region] = region_embed

        # Cross-region fusion with anatomical constraints
        region_embeds = self._anatomically_constrained_fusion(region_embeds)

        return region_embeds

        # Process each brain region
        for region, info in self.regions.items():
            # Get text embeddings for this region's levels
            region_text = torch.stack([
                text_embeds[level].mean(0) for level in
                'paragraphs': 0.6  # Lowest for highest level
            },
            'loss_scaling': {
                'enabled': True,
                'scale_factor': 1e4,  # Initial loss scale
                'growth_factor': 2.0,  # Scale growth rate
                'backoff_factor': 0.5,  # Scale reduction rate
                'growth_interval': 2000  # Steps between scale increases
            },
            'weights': {
                'enabled': True,
                'ema_decay': 0.99,  # Exponential moving average decay
                'update_freq': 100,  # Steps between weight updates
                'min_weight': 0.1,   # Minimum weight value
                'max_weight': 10.0   # Maximum weight value
            }
        },
        evaluation_metrics: Optional[Dict[str, Any]] = {
            'hierarchical_metrics': {
                'level_specific': {
                    'bytes': ['accuracy', 'perplexity', 'entropy', 'compression_ratio'],
                    'chars': ['accuracy', 'perplexity', 'vocab_coverage', 'char_error_rate'],
                    'words': ['accuracy', 'perplexity', 'bleu', 'semantic_similarity'],
                    'sentences': ['accuracy', 'perplexity', 'rouge', 'syntactic_complexity'],
                    'paragraphs': ['accuracy', 'perplexity', 'coherence', 'topic_diversity']
                },
                'cross_level': {
                    'abstraction_ratio': True,      # Measure feature abstraction between levels
                    'information_flow': True,       # Track information propagation
                    'alignment_score': True,        # Measure hierarchical alignment
                    'compression_efficiency': True  # Evaluate information compression
                },
                'brain_metrics': {
                    'region_accuracy': True,        # Per-region prediction accuracy
                    'activation_patterns': True,    # Brain activation pattern analysis
                    'temporal_correlation': True,   # Temporal correlation with EEG/fMRI
                    'spatial_correlation': True     # Spatial correlation with brain regions
                },
                'learning_dynamics': {
                    'curriculum_progress': True,    # Track curriculum learning progress
                    'knowledge_retention': True,    # Measure knowledge retention
                    'transfer_efficiency': True,    # Evaluate transfer between levels
                    'adaptation_speed': True        # Monitor adaptation to new levels
                }
            },
            'tracking_config': {
                'moving_average': 100,              # Window for moving averages
                'log_frequency': 10,                # Steps between logging
                'detailed_analysis': 1000,          # Steps between detailed analysis
                'save_distributions': True,         # Save score distributions
                'plot_learning_curves': True        # Generate learning curves
            },
            'threshold_config': {
                'min_accuracy': {
                    'bytes': 0.7,
                    'chars': 0.65,
                    'words': 0.6,
                    'sentences': 0.55,
                    'paragraphs': 0.5
                },
                'max_perplexity': {
                    'bytes': 50,
                    'chars': 100,
                    'words': 200,
                    'sentences': 300,
                    'paragraphs': 400
                },
                'min_correlation': 0.5,             # Minimum brain correlation
                'max_error_rate': 0.3               # Maximum error rate
            }
        },

        visualization_config: Optional[Dict[str, Any]] = {
            'enabled': True,
            'plot_types': {
                'learning_curves': {
                    'metrics': ['loss', 'accuracy', 'perplexity'],
                    'per_level': True,
                    'smoothing': 0.9,
                    'update_freq': 100
                },
                'attention_maps': {
                    'enabled': True,
                    'max_heads': 4,
                    'levels': ['bytes', 'chars', 'words', 'sentences', 'paragraphs'],
                    'update_freq': 500
                },
                'feature_spaces': {
                    'enabled': True,
                    'method': 'tsne',  # or 'umap', 'pca'
                    'n_samples': 1000,
                    'update_freq': 1000
                },
                'brain_activations': {
                    'enabled': True,
                    'regions': [
                        'visual',
                        'language', 
                        'memory',
                        'attention',
                        'executive',
                        'semantic',
                        'integration',
                        'sensory_temporal',
                        'sensory_parietal'
                    ],
                    'plot_type': 'heatmap',
                    'update_freq': 500
                }
            },
            'hierarchical_viz': {
                'level_transitions': {
                    'enabled': True,
                    'show_connections': True,
                    'update_freq': 200
                },
                'feature_evolution': {
                    'enabled': True,
                    'track_changes': True,
                    'update_freq': 500
                },
                'attention_flow': {
                    'enabled': True,
                    'show_weights': True,
                    'update_freq': 300
                }
            },
            'interactive_plots': {
                'enabled': True,
                'server_port': 8050,
                'live_updates': True
            },
            'export_config': {
                'save_plots': True,
                'format': 'png',
                'dpi': 300,
                'path': 'visualizations'
            }
        },
        attention_config: Optional[Dict[str, Any]] = {
            'num_heads': {
                'bytes': 8,       # More heads for fine-grained attention
                'chars': 8,
                'words': 6,       # Balanced attention
                'sentences': 4,    # Fewer heads for high-level patterns
                'paragraphs': 4
            },
            'head_dim': 64,       # Dimension per attention head
            'dropout': 0.1,       # Attention dropout
            'max_relative_pos': {  # Maximum relative position per level
                'bytes': 128,
                'chars': 64,
                'words': 32,
                'sentences': 16,
                'paragraphs': 8
            },
            'cross_level_attention': True,  # Enable cross-level attention
            'hierarchical_pos_encoding': True,  # Enable hierarchical position encoding
            'attention_pruning': True,  # Enable dynamic attention pruning
            'routing_temperature': 0.1,  # Temperature for attention routing
        },
        level_loss_weights: Optional[Dict[str, float]] = {
            'bytes': 1.0,      # Base level weight
            'chars': 1.2,      # Slightly higher for character understanding
            'words': 1.5,      # Important for semantic meaning
            'sentences': 1.8,  # Critical for context
            'paragraphs': 2.0  # Highest for global understanding
        },
        regularization_config: Optional[Dict[str, Any]] = {
            'weights': {
                'abstraction': 0.1,     # Progressive feature abstraction
                'bottleneck': 0.05,     # Information bottleneck
                'consistency': 0.15,    # Hierarchical consistency
                'sparsity': 0.1,       # Adaptive sparsity
                'attention': 0.1,      # Attention regularization
                'orthogonality': 0.05  # Level orthogonality
            },
            'structural_constraints': {
                'feature_hierarchy': {
                    'enabled': True,
                    'weight': 0.1,
                    'min_ratio': 1.2,  # Minimum abstraction ratio between levels
                    'max_overlap': 0.5  # Maximum feature overlap between levels
                },
                'information_flow': {
                    'enabled': True,
                    'weight': 0.1,
                    'bottleneck_factor': 0.8,  # Information compression factor
                    'skip_penalty': 0.2   # Penalty for skipping levels
                },
                'representation_structure': {
                    'enabled': True,
                    'weight': 0.1,
                    'sparsity_schedule': {
                        'bytes': 0.1,      # Dense representations
                        'chars': 0.2,
                        'words': 0.3,
                        'sentences': 0.4,
                        'paragraphs': 0.5   # Sparse representations
                    },
                    'group_sparsity': True  # Enable group sparsity
                },
                'attention_patterns': {
                    'enabled': True,
                    'weight': 0.1,
                    'local_attention': {    # Local attention weights
                        'bytes': 0.8,       # Strong local attention
                        'chars': 0.6,
                        'words': 0.4,
                        'sentences': 0.2,
                        'paragraphs': 0.1   # Weak local attention
                    },
                    'cross_level': 0.05     # Cross-level attention weight
                },
                'temporal_coherence': {
                    'enabled': True,
                    'weight': 0.1,
                    'sequence_length': {    # Sequence length ratios
                        'bytes_to_chars': 4,
                        'chars_to_words': 5,
                        'words_to_sentences': 10,
                        'sentences_to_paragraphs': 5
                    },
                    'smoothness': 0.1       # Temporal smoothness weight
                }
            },
            'regularization_schedule': {
                'warmup_steps': 1000,       # Steps before full regularization
                'decay_rate': 0.99,         # Exponential decay rate
                'min_weight': 0.01,         # Minimum regularization weight
                'update_freq': 100          # Update frequency in steps
            },
            'level_specific': {
                'bytes': {
                    'abstraction_weight': 0.05,
                    'sparsity_target': 0.1,
                    'attention_temp': 0.1
                },
                'chars': {
                    'abstraction_weight': 0.1,
                    'sparsity_target': 0.2,
                    'attention_temp': 0.2
                },
                'words': {
                    'abstraction_weight': 0.15,
                    'sparsity_target': 0.3,
                    'attention_temp': 0.3
                },
                'sentences': {
                    'abstraction_weight': 0.2,
                    'sparsity_target': 0.4,
                    'attention_temp': 0.4
                },
                'paragraphs': {
                    'abstraction_weight': 0.25,
                    'sparsity_target': 0.5,
                    'attention_temp': 0.5
                }
            }
        },
        level_reg_scales: Optional[Dict[str, float]] = {
            'bytes': 0.5,      # Lower regularization for basic features
            'chars': 0.7,      # Gradually increase regularization
            'words': 1.0,      # Standard regularization
            'sentences': 1.3,  # Higher regularization for structure
            'paragraphs': 1.5  # Strongest regularization for high-level
        },
        dynamic_weight_schedule: Optional[Dict[str, List[float]]] = {
            'bytes': [1.0, 0.8, 0.6, 0.4, 0.2],      # Decrease over time
            'chars': [0.2, 1.0, 0.8, 0.6, 0.4],      # Peak early
            'words': [0.2, 0.4, 1.0, 0.8, 0.6],      # Peak mid-training
            'sentences': [0.2, 0.4, 0.6, 1.0, 0.8],  # Peak late
            'paragraphs': [0.2, 0.4, 0.6, 0.8, 1.0]  # Increase over time
        },
        curriculum_warmup: int = 1000,  # Steps to warm up each level
        curriculum_overlap: float = 0.1,  # Overlap between levels
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        save_every: int = 1000,
        eval_every: int = 100,
        patience: int = 10,  # Early stopping patience
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = True,
        wandb_project: str = "brain-aware-blt",
        device: Optional[str] = None,
        modalities: List[str] = ['text'], #Removed EEG and fMRI
        d_model: int = 512,
        n_layers: int = 24,
        n_heads: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 9,
        window_size: int = 512,
        max_ngram: int = 8,
        hash_vocab_size: int = 300000,
        dropout: float = 0.1,
        paragraph_dim: int = 1024,
        region_dim: int = 256,  # Dimension for brain region embeddings
        
        # Entropy-based patching parameters
        entropy_threshold: float = 0.5,  # Global entropy threshold
        relative_threshold: float = 0.2,  # Relative entropy threshold
        min_patch_size: int = 4,  # Minimum patch size in bytes
        max_patch_size: int = 32,  # Maximum patch size in bytes
        
        # SELF-GOAL parameters
        max_goals: int = 5,
        min_importance: float = 0.1,
        exploration_factor: float = 0.5,
        initial_temperature: float = 1.0
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.save_every = save_every
        self.eval_every = eval_every
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modalities = modalities
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.window_size = window_size
        self.max_ngram = max_ngram
        self.hash_vocab_size = hash_vocab_size
        self.dropout = dropout
        self.paragraph_dim = paragraph_dim
        self.region_dim = region_dim
        
        # SELF-GOAL parameters
        self.max_goals = max_goals
        self.min_importance = min_importance
        self.exploration_factor = exploration_factor
        self.initial_temperature = initial_temperature

        # Entropy-based patching parameters
        self.entropy_threshold = entropy_threshold
        self.relative_threshold = relative_threshold
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

#start the background training for the clustering episodic memory nuerogensis. 
trainer = BrainAwareBLTTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
trainer.model.episodic_memory.start_clustering_thread()

# Updated BrainAwareBLTTrainer with HierarchicalEpisodicMemory
class BrainAwareBLTTrainer:
    """Trainer for brain-aware BLT model"""
    def __init__(
        self,
        model: 'ByteLatentTransformer',
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional['TrainingConfig'] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        # Initialize HierarchicalEpisodicMemory
        episodic_memory_config = EpisodicMemoryConfig(
            capacity=self.config.episodic_memory_capacity,
            cluster_method=self.config.episodic_memory_cluster_method,
            hdbscan_min_cluster_size=self.config.hdbscan_min_cluster_size,
            hdbscan_metric=self.config.hdbscan_metric,
            hdbscan_cluster_selection_epsilon=self.config.hdbscan_cluster_selection_epsilon,
            som_grid_size=self.config.som_grid_size,
            som_input_dim=self.config.som_input_dim,
            som_sigma=self.config.som_sigma,
            som_learning_rate=self.config.som_learning_rate,
            enable_cluster_maintenance=self.config.enable_cluster_maintenance,
            cluster_maintenance_interval=self.config.cluster_maintenance_interval,
            cluster_centroid_update_freq=self.config.cluster_centroid_update_freq,
            cluster_relevance_threshold=self.config.cluster_relevance_threshold,
            max_cluster_size=self.config.max_cluster_size,
            merge_similarity_threshold=self.config.merge_similarity_threshold,
            hierarchical_clustering_method=self.config.hierarchical_clustering_method
        )
        self.model.episodic_memory = HierarchicalEpisodicMemory(
            capacity=self.config.episodic_memory_capacity,
            config=episodic_memory_config
        )

        # Add these lines to store optimizer and scheduler state dicts
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

        # Add variables to track self-evolution
        self.current_round = 0  # Start with round 0
        self.model.goal_manager.set_model(self.model)

        # Move model to device
        self.model = self.model.to(self.config.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

def train(self) -> None:
      """
      Train the model using the rStar-Math self-evolution process.
      """
              for round_num in range(1, self.config.r_star_rounds + 1):
              self.current_round = round_num
              print(f"Starting rStar-Math self-evolution round {round_num}...")

              # 1. Data Generation (with MCTS and Subgoal Management)
              train_data = self.generate_training_data()
              
              # Compute embeddings for clustering evaluation
              if train_data:
                  embeddings = [torch.cat([ex['goal'], ex['context']], dim=-1).cpu().detach().numpy() for ex in train_data]
                  embeddings = np.array(embeddings)

                  # Evaluate clustering quality
                  clustering_metrics = self.model.episodic_memory.evaluate_clustering(embeddings)

                  # Log clustering metrics
                  if self.config.use_wandb:
                      for metric_name, metric_value in clustering_metrics.items():
                          wandb.log({f"train/clustering/{metric_name}": metric_value, "round": round_num})
              else:
                  print("No training data generated for clustering evaluation.")

              # 2. Train Policy Model (SLM)
              print(f"Fine-tuning policy model (round {round_num})...")
              self.fine_tune_policy_model(train_data)

              # 3. Train Process Preference Model (PPM)
              print(f"Training process preference model (round {round_num})...")
              self.train_process_preference_model(train_data)

              # 4. Evaluate and Log
              if self.val_loader is not None:
                  print(f"Evaluating model (round {round_num})...")
                  val_metrics = self._validate()
                  # Log the validation metrics using wandb or other logging mechanism
                  if self.config.use_wandb:
                      wandb.log({f"val/round_{round_num}/{k}": v for k, v in val_metrics.items()})

              # 5. Save Checkpoint (optional)
              self.save_checkpoint(f'checkpoint_round_{round_num}.pt')

          print("rStar-Math self-evolution completed.")

def generate_training_data(self) -> List[Dict]:
        """
        Generates training data using MCTS with step-by-step verification and subgoal management.
        """
        generated_data = []

        # Update MCTS parameters for this round
        if self.current_round == 1:
          self.model.set_mcts_params(
            rollouts=self.config.initial_mcts_rollouts,
            expansion_factor=self.config.mcts_expansion_factor,
            c=self.config.mcts_c
          )
        elif self.current_round < self.config.r_star_rounds:
          self.model.set_mcts_params(
            rollouts=self.config.initial_mcts_rollouts,
            expansion_factor=self.config.mcts_expansion_factor,
            c=self.config.mcts_c
          )
        else:
          self.model.set_mcts_params(
            rollouts=self.config.increased_mcts_rollouts,
            expansion_factor=self.config.mcts_expansion_factor,
            c=self.config.mcts_c
          )

        # Set the temperature for goal selection in GoalManager (for exploration-exploitation balance)
        self.model.set_temperature(self.config.initial_temperature)

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Generating data (round {self.current_round})")):
            # Move batch to device
            batch = {
                k: {
                    k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                }
                for k, v in batch.items()
            }

            # Extract the main goal from the input batch
            main_goal = self.model._extract_high_level_goal(batch['tokens']['bytes'])

            # Decompose the main goal into subgoals and select relevant ones
            context = {
                'input': batch['tokens']['bytes'],  # Using byte-level tokens as context
                'brain_regions': batch.get('region_embeds'),  # Include brain region data if available
                'step': self.global_step  # You can use the global step as part of the context
            }
            subgoals = self.model._decompose_and_select_subgoals(main_goal, context)

            # Execute subgoals with Chain-of-Thought and get the final output and reasoning steps
            final_output, reasoning_steps, final_reward = self.model._execute_with_chain_of_thought(subgoals, batch['tokens']['bytes'])
            
            # Store the experience in episodic memory
            experience = {
                'goal': main_goal,
                'context': context,
                'subgoals': subgoals,
                'reasoning_steps': reasoning_steps,
                'final_reward': final_reward
            }
            self.model.store_experience(experience)

            # Update the goal tree based on the final reward
            self.model.goal_manager.update_tree(final_reward)

            # Here you would convert the reasoning steps into a suitable format for training
            # For example, format the steps into a sequence of text for SFT
            for step in reasoning_steps:
                # Format the step into training data
                # This is a placeholder for how you might convert a step into a training example
                example = self.format_reasoning_step_as_training_example(step)
                generated_data.append(example)

        return generated_data
    def format_reasoning_step_as_training_example(self, step: Dict) -> Dict:
        """
        Formats a reasoning step from the MCTS into a training example.
        This is a placeholder function that you'll need to adapt based on how you want to
        structure your training data.
        """
        # Example: Convert a step into an input-output pair for supervised fine-tuning
        input_text = f"Goal: {step['goal']}\nContext: {step['input']}\n"
        
        if 'output' in step:
            output_text = f" -> {step['output']}"  # The actual reasoning step taken
        elif 'reflection' in step:
            output_text = f" -> Reflection: {step['reflection']}"
        else:
            output_text = ""

        return {
            'input': input_text,
            'output': output_text,
            'reward': step.get('reward', 0.0)  # You might have a reward associated with the step
        }

    def fine_tune_policy_model(self, train_data: List[Dict]):
        """
        Fine-tunes the policy model using the generated training data.
        """
        # Convert the generated data into a suitable dataset format
        dataset = self.create_dataset_from_examples(train_data)

        # Create a DataLoader for the dataset
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )


    def visualize_activations_3d(self, activation_dict, region_mapping, method='pca', output_filename=None):
        """
        Visualizes model activations in 3D using dimensionality reduction.

        Args:
            activation_dict: Dictionary containing activations from different layers/modules.
            region_mapping: Dictionary mapping region names to layer names.
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap').
            output_filename: Optional filename to save the plot.
        """
        all_activations = []
        all_regions = []

        for region, layer_names in region_mapping.items():
            for layer_name in layer_names:
                if layer_name in activation_dict:
                    # Flatten activations and append
                    activations = activation_dict[layer_name].flatten().cpu().numpy()
                    all_activations.append(activations)
                    all_regions.extend([region] * len(activations))

        if not all_activations:
            print("No activations found for visualization.")
            return

        all_activations = np.stack(all_activations)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=3)
        elif method == 'tsne':
            reducer = TSNE(n_components=3)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=3)
        else:
            raise ValueError(f"Invalid dimensionality reduction method: {method}")

        reduced_data = reducer.fit_transform(all_activations)

        # Create a 3D scatter plot using Plotly
        fig = go.Figure()

        for region in set(all_regions):
            indices = [i for i, r in enumerate(all_regions) if r == region]
            fig.add_trace(go.Scatter3d(
                x=reduced_data[indices, 0],
                y=reduced_data[indices, 1],
                z=reduced_data[indices, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.8),
                name=region
            ))

        fig.update_layout(title="3D Visualization of Brain Region Activations")

        if output_filename:
            fig.write_html(output_filename)  # Save as interactive HTML
        else:
            fig.show()

        # Set up a new optimizer and scheduler for fine-tuning
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate / 10)  # Lower LR for fine-tuning
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # Example scheduler

        # Fine-tuning loop
        self.model.train()
        for epoch in range(self.config.num_epochs):
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Fine-tuning policy model (epoch {epoch+1})")):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)  # Adapt this based on your model's forward method

                # Compute loss
                loss = self._compute_loss(outputs, batch)  # You might need a specialized loss function

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Log metrics
                if self.config.use_wandb:
                    wandb.log({
                        f"train/fine_tune/round_{self.current_round}/loss": loss.item(),
                        f"train/fine_tune/round_{self.current_round}/epoch": epoch,
                        f"train/fine_tune/round_{self.current_round}/step": self.global_step
                    })

                self.global_step += 1

    def create_dataset_from_examples(self, examples: List[Dict]) -> Dataset:
        """
        Creates a PyTorch Dataset from a list of training examples.
        You'll need to implement this based on the structure of your training examples.
        """
        # This is a placeholder function. You'll need to implement the actual logic
        # to convert your examples into a format that can be used by a PyTorch Dataset.

        # Example: Assuming your examples are dictionaries with 'input' and 'output' keys
        inputs = [ex['input'] for ex in examples]
        outputs = [ex['output'] for ex in examples]

        # Tokenize the inputs and outputs
        tokenized_inputs = self.config.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        tokenized_outputs = self.config.tokenizer(outputs, padding=True, truncation=True, return_tensors="pt")

        # Create a TensorDataset
        dataset = torch.utils.data.TensorDataset(
            tokenized_inputs['input_ids'],
            tokenized_inputs['attention_mask'],
            tokenized_outputs['input_ids'],
            tokenized_outputs['attention_mask']
        )

        return dataset

    def _compute_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Computes the loss for fine-tuning the policy model.
        You'll need to implement this based on your model's outputs and the structure of your training data.
        """
        # This is a placeholder function. You might need to use a different loss function
        # depending on how you've structured your training data.

        # Example: Assuming a sequence-to-sequence setup with a language modeling objective
        logits = outputs['logits']  # Assuming your model outputs logits
        labels = batch['output_ids']  # Assuming you have a 'output_ids' tensor in your batch

        # Flatten the logits and labels for the CrossEntropyLoss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss

    def train_process_preference_model(self, train_data: List[Dict]):
        """
        Trains the Process Preference Model (PPM) using the generated training data.
        """
        # Convert the generated data into a suitable dataset format for PPM training
        # You might need to create pairs of trajectories or steps for preference learning
        dataset = self.create_ppm_dataset_from_examples(train_data)

        # Create a DataLoader for the dataset
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # Assuming you have a separate optimizer for the PPM
        optimizer = optim.AdamW(self.model.process_preference_model.parameters(), lr=self.config.learning_rate / 10)  # Lower LR for PPM
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        # PPM training loop
        self.model.process_preference_model.train()
        for epoch in range(self.config.num_epochs):
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Training PPM (epoch {epoch+1})")):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass through the PPM
                outputs = self.model.process_preference_model(batch)

                # Compute the PPM loss (e.g., pairwise ranking loss)
                loss = self._compute_ppm_loss(outputs, batch)  # Implement this based on your PPM

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Log metrics
                if self.config.use_wandb:
                    wandb.log({
                        f"train/ppm/round_{self.current_round}/loss": loss.item(),
                        f"train/ppm/round_{self.current_round}/epoch": epoch,
                        f"train/ppm/round_{self.current_round}/step": self.global_step
                    })

                self.global_step += 1

    def create_ppm_dataset_from_examples(self, examples: List[Dict]) -> Dataset:
        """
        Creates a dataset suitable for training the Process Preference Model (PPM).
        This might involve creating pairs of trajectories or steps for preference learning.
        """
        # Placeholder for creating a dataset for PPM training
        # You need to adapt this based on how your PPM is designed and what kind of input it expects

        # Example: Create preference pairs from trajectories with different rewards
        preference_pairs = []
        for example in examples:
            if len(example['reasoning_steps']) >= 2:  # Need at least 2 steps to create a pair
                # This is a simplified example. You might want to use more sophisticated criteria
                # for selecting pairs based on rewards, confidence scores, etc.
                step1 = example['reasoning_steps'][0]
                step2 = example['reasoning_steps'][-1]  # Compare the first and last steps

                if step1['reward'] > step2['reward']:
                    preference_pairs.append((step1, step2))  # step1 is preferred over step2
                elif step2['reward'] > step1['reward']:
                    preference_pairs.append((step2, step1))  # step2 is preferred over step1

        # Convert the preference pairs into a format that can be used by a PyTorch Dataset
        # You might need to tokenize the steps, pad sequences, etc.
        inputs = [self.config.tokenizer(pair[0]['input'], pair[0]['output'], padding=True, truncation=True, return_tensors="pt") for pair in preference_pairs]
        labels = [1 if pair[0]['reward'] > pair[1]['reward'] else 0 for pair in preference_pairs]  # Binary preference labels

        # Create a TensorDataset
        dataset = torch.utils.data.TensorDataset(
            torch.stack([inp['input_ids'].squeeze() for inp in inputs]),
            torch.stack([inp['attention_mask'].squeeze() for inp in inputs]),
            torch.tensor(labels, dtype=torch.long)
        )

        return dataset

    def _compute_ppm_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Computes the loss for training the Process Preference Model (PPM).
        This is a placeholder function. You'll need to implement the actual loss calculation
        based on your PPM's architecture and training objective.
        """
        # Example: Pairwise ranking loss (e.g., from the "Learning to Rank" literature)
        # Assuming your PPM outputs a score for each input and the batch contains pairs of inputs
        # where the first element is preferred over the second

        scores = outputs['scores']  # Assuming your PPM outputs a 'scores' tensor
        labels = batch['labels']  # Assuming you have a 'labels' tensor indicating preference (e.g., 1 for first element preferred, 0 otherwise)

        # Calculate the pairwise ranking loss
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return loss
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    'model_config': {
                        'd_model': self.config.d_model,
                        'n_layers': self.config.n_layers,
                        'n_heads': self.config.n_heads,
                        'encoder_layers': self.config.encoder_layers,
                        'decoder_layers': self.config.decoder_layers,
                        'window_size': self.config.window_size,
                        'max_ngram': self.config.max_ngram,
                        'hash_vocab_size': self.config.hash_vocab_size,
                        'dropout': self.config.dropout,
                        'paragraph_dim': self.config.paragraph_dim,
                        'modalities': self.config.modalities,
                        'region_dim': self.config.region_dim
                    },
                    'train_config': self.config.__dict__
                }
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=len(self.train_loader) * self.config.num_epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    def train(self) -> None:
        """Train model with dynamic head pruning and growth"""
        self.model.train()
        
        # Track architecture changes
        pruned_heads = set()
        added_heads = set()
        architecture_history = []
        
        # Initialize head utilization tracking
        head_utilization = {}
        complexity_metrics = []
        
        for epoch in range(self.config.num_epochs):
            # Consider pruning after initial training period
            if epoch > self.config.num_epochs * 0.2:  # Start pruning after 20% of training
                pruning_candidates = self._identify_pruning_candidates()
                
                if pruning_candidates:
                    # Validate pruning impact
                    original_val_metrics = self._validate() if self.val_loader else None
                    
                    for head_info in pruning_candidates:
                        if self._validate_head_pruning(head_info, original_val_metrics):
                            self._prune_head(head_info['head'])
                            pruned_heads.add(head_info['head'])
                            pruning_history.append({
                                'epoch': epoch,
                                'head': head_info['head'],
                                'redundancy_score': head_info['redundancy_score'],
                                'impact_score': head_info['impact_score']
                            })
                            
                            # Log pruning event
                            if self.config.use_wandb:
                                wandb.log({
                                    'pruning/pruned_head': head_info['head'],
                                    'pruning/total_pruned': len(pruned_heads),
                                    'pruning/epoch': epoch
                                })
            self._train_epoch(epoch)
            
            if self.val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self._validate()
                val_loss = val_metrics['loss']
                
                # Check for early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best.pt')
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.patience:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
    
    def _train_epoch(
        self,
        epoch: int
    ) -> None:
        """Train one epoch"""
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                metrics = self._train_step(batch)
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_perplexity += metrics['perplexity']
                
                # Update progress bar
                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                
                postfix = {
                    'loss': f'{metrics["loss"]:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}',
                    'avg_acc': f'{total_accuracy/(batch_idx+1):.4f}',
                    'ppl': f'{metrics["perplexity"]:.4f}',
                    'avg_ppl': f'{total_perplexity/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}',
                    
                    # Hierarchical metrics
                    'hier': f'{metrics["hierarchical_accuracy"]:.4f}',
                    'byte': f'{metrics.get("bytes_accuracy", 0):.4f}',
                    'word': f'{metrics.get("words_accuracy", 0):.4f}',
                    'sent': f'{metrics.get("sentences_accuracy", 0):.4f}',
                    
                    # Patch metrics
                    'patch': f'{metrics.get("patch_accuracy", 0):.4f}',
                    
                    # Brain region metrics
                    'brain': f'{metrics.get("brain_region_accuracy", 0):.4f}',
                    'vis': f'{metrics.get("visual_accuracy", 0):.4f}',
                    'lang': f'{metrics.get("language_accuracy", 0):.4f}',
                    'mem': f'{metrics.get("memory_accuracy", 0):.4f}',
                    'attn': f'{metrics.get("attention_accuracy", 0):.4f}',
                    'exec': f'{metrics.get("executive_accuracy", 0):.4f}',
                    'sem': f'{metrics.get("semantic_accuracy", 0):.4f}',
                    'int': f'{metrics.get("integration_accuracy", 0):.4f}',
                    'temp': f'{metrics.get("sensory_temporal_accuracy", 0):.4f}',
                    'par': f'{metrics.get("sensory_parietal_accuracy", 0):.4f}'
                }
                pbar.set_postfix(postfix)
                
                # Log metrics
                if self.config.use_wandb:
                    # Prepare gradient and parameter histograms
                    histograms = {}
                    for name, param in self.model.named_parameters():
                        # Get layer name
                        layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
                        
                        # Parameter histogram
                        histograms[f'parameters/{layer_name}_hist'] = wandb.Histogram(
                            param.detach().cpu().numpy()
                        )
                        
                        # Parameter statistics
                        with torch.no_grad():
                            param_mean = param.mean().item()
                            param_std = param.std().item()
                            param_min = param.min().item()
                            param_max = param.max().item()
                            param_norm = torch.norm(param).item()
                            
                            histograms.update({
                                f'parameters/{layer_name}_mean': param_mean,
                                f'parameters/{layer_name}_std': param_std,
                                f'parameters/{layer_name}_min': param_min,
                                f'parameters/{layer_name}_max': param_max,
                                f'parameters/{layer_name}_norm': param_norm
                            })
                        
                        # Gradient histogram
                        if param.grad is not None:
                            histograms[f'gradients/{layer_name}_hist'] = wandb.Histogram(
                                param.grad.detach().cpu().numpy()
                            )
                    
                    # Log metrics and histograms
                    wandb.log({
                        # Overall metrics
                        'train/loss': metrics['loss'],
                        'train/hierarchical_accuracy': metrics['hierarchical_accuracy'],
                        'train/patch_accuracy': metrics.get('patch_accuracy', 0),
                        'train/combined_accuracy': metrics['accuracy'],
                        'train/perplexity': metrics['perplexity'],
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                        
                        # Brain region accuracies
                        'train/visual_accuracy': metrics.get('visual_accuracy', 0),
                        'train/language_accuracy': metrics.get('language_accuracy', 0),
                        'train/memory_accuracy': metrics.get('memory_accuracy', 0),
                        'train/attention_accuracy': metrics.get('attention_accuracy', 0),
                        'train/executive_accuracy': metrics.get('executive_accuracy', 0),
                        'train/semantic_accuracy': metrics.get('semantic_accuracy', 0),
                        'train/integration_accuracy': metrics.get('integration_accuracy', 0),
                        'train/sensory_temporal_accuracy': metrics.get('sensory_temporal_accuracy', 0),
                        'train/sensory_parietal_accuracy': metrics.get('sensory_parietal_accuracy', 0),
                        
                        # Brain region MSE
                        'train/visual_mse': metrics.get('visual_mse', 0),
                        'train/language_mse': metrics.get('language_mse', 0),
                        'train/memory_mse': metrics.get('memory_mse', 0),
                        'train/attention_mse': metrics.get('attention_mse', 0),
                        'train/executive_mse': metrics.get('executive_mse', 0),
                        'train/semantic_mse': metrics.get('semantic_mse', 0),
                        'train/integration_mse': metrics.get('integration_mse', 0),
                        'train/sensory_temporal_mse': metrics.get('sensory_temporal_mse', 0),
                        'train/sensory_parietal_mse': metrics.get('sensory_parietal_mse', 0),
                        
                        # Overall gradient metrics
                        'train/gradient_norm': metrics['grad_norm'],
                        'train/gradient_norm_before_clip': metrics['grad_norm_before_clip'],
                        'train/gradient_norm_after_clip': metrics['grad_norm_after_clip'],
                        'train/gradient_mean': metrics['grad_mean'],
                        'train/gradient_std': metrics['grad_std'],
                        'train/gradient_min': metrics['grad_min'],
                        'train/gradient_max': metrics['grad_max'],
                        'train/gradient_clip_ratio': metrics['grad_clip_ratio'],
                        
                        # Per-layer gradient statistics
                        **{f'train/gradients/{k}': v 
                           for k, v in metrics.items() 
                           if k.startswith('grad_') and k not in [
                               'grad_norm', 'grad_norm_before_clip', 'grad_norm_after_clip',
                               'grad_mean', 'grad_std', 'grad_min', 'grad_max', 'grad_clip_ratio'
                           ]},
                           
                        # Gradient histograms
                        **histograms
                    })
                
                self.global_step += 1
    
    def _classify_fact_based(self, instruction: str) -> bool:
        """Classify if instruction requires factual response"""
        # Use SFT model to classify
        prompt = f"Here is a question from a user: \"{instruction}\". To answer the above question, do you need the factual knowledge from Wikipedia? Give an answer using the format: \"Answer: Yes or No\"."
        
        with torch.no_grad():
            # Get model response
            response = self.model.generate(
                prompt,
                max_length=100,
                temperature=0.1
            )
            
            # Extract Yes/No answer
            is_fact_based = "yes" in response.lower()
            
        return is_fact_based

    def _train_step(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Train one step"""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {
            k: {
                k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
                for k2, v2 in v.items()
            }
            for k, v in batch.items()
        }

def _train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Train one step with episodic memory updates and brain region-specific processing."""
    self.optimizer.zero_grad()

    # Move batch to device
    batch = {
        k: {
            k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
            for k2, v2 in v.items()
        }
        for k, v in batch.items()
    }

 # Forward pass with activation collection
    activation_dict = {} # The audio, image, fmri, audio, decoder with the text being below. It needs to the properly integrated into the training step with placeholder removed. 
    attention_patterns = {}
    hooks = []
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._hook_fn(mod, inp, out, name, activation_dict, attention_patterns)
                )
            )

    # Classify if instruction is fact-based
    instruction = batch['instruction'][0] if isinstance(batch['instruction'], list) else batch['instruction']
    self.model.is_fact_based = self._classify_fact_based(instruction)

    # Forward pass with all data modalities
    outputs = self.model(
        batch['tokens']['bytes'],
        goal_context=None,
        brain_regions=batch.get('region_embeds'),
        rnn_states=None,
        sensory_data=batch.get('sensory_data'),
        tactile_data=batch.get('tactile_data'),
        audio_data=batch.get('audio_data'),
        image_data=batch.get('image_data')
    )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute hierarchical loss
    loss = self._compute_hierarchical_loss(outputs, batch)

    # Add specific loss for sensory/tactile training if needed
    if self.current_round == 'sensory_tactile_training':
        loss += self._compute_sensory_tactile_loss(outputs, batch)

    # Backward pass
    loss.backward()

    # Calculate per-layer gradient statistics
    layer_grads = {}
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
            grad = param.grad.detach()
            layer_grads[layer_name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': torch.norm(grad).item(),
                'sparsity': (grad == 0).float().mean().item()
            }

            # Gradient clipping
            if self.config.gradient_clip > 0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                grad_norm_after = torch.norm(param.grad.detach())

                metrics = {
                    'grad_norm_before_clip': grad_norm_before.item(),
                    'grad_norm_after_clip': grad_norm_after.item(),
                    'grad_clip_ratio': grad_norm_after.item() / (grad_norm_before.item() + 1e-8)
                }

                metrics.update({
                    f'grad_{k}': v for k, v in layer_grads.items()
                })

    # Step optimizer
    self.optimizer.step()
    self.scheduler.step()

    # Update episodic memory
    if self.config.use_episodic_memory:
        main_goal = self._extract_high_level_goal(batch['tokens']['bytes'])
        context = {
            'input': batch['tokens']['bytes'],
            'brain_regions': batch.get('region_embeds'),
            'step': self.global_step
        }
        subgoals = self.model._decompose_and_select_subgoals(main_goal, context)

        if 'reasoning_steps' in outputs and outputs['reasoning_steps']:
            final_output, reasoning_steps, final_reward = self.model._execute_with_chain_of_thought(subgoals, batch['tokens']['bytes'])

            self.model.store_experience(main_goal, context, subgoals, reasoning_steps, final_reward)
            self.model.goal_manager.update_tree(final_reward)
            retrieved_memories = self.model.retrieve_memories(main_goal, context, num_memories=self.config.num_retrieve_memories)
            self.model.sparse_update(retrieved_memories)
            self.model.update_memory_importance()

            if self.global_step % self.config.expansion_check_interval == 0:
                self.model.monitor_activations('subgoal_generator')
        else:
            print("Warning: No reasoning steps found in model output.")

    # Compute metrics
    metrics = self._compute_metrics(outputs, batch, loss.item())
    metrics.update({
        'grad_norm': grad_norm_before.item() if 'grad_norm_before_clip' in locals() else 0.0,
        'grad_mean': torch.mean(torch.stack([
            param.grad.mean()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_std': torch.std(torch.stack([
            param.grad.std()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_min': torch.min(torch.stack([
            param.grad.min()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_max': torch.max(torch.stack([
            param.grad.max()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item()
    })

    # Add the hook function to the class
    def _hook_fn(self, module, input, output, name, activation_dict, attention_patterns):
        # Your existing hook logic
        pass

    # Add the compute_loss_from_memory method to the class
    def compute_loss_from_memory(self, memory):
      """
      Computes a loss based on a retrieved memory.
      This is a placeholder that you need to adapt based on the nature of your memories and your training objectives.
      """
      if memory['type'] == 'subgoal_generation':
          # Example: MSE loss between generated subgoal and memory subgoal
          generated_subgoal = self.subgoal_generator(torch.cat([memory['goal'], memory['context']], dim=-1))
          target_subgoal = memory['subgoal']
          loss = nn.MSELoss()(generated_subgoal, target_subgoal)

      elif memory['type'] == 'action_selection':
          # Example: Cross-entropy loss for action prediction
          # This requires adapting the model to output action probabilities
          # and having a way to compute the "correct" action from the memory
          logits = self.model(memory['context'])  # Assuming your model can take context and produce logits
          target_action = memory['action']  # This would need to be the index of the correct action
          loss = F.cross_entropy(logits, target_action)

      else:
          raise ValueError(f"Unknown memory type: {memory['type']}")

      return loss

    return metrics

    # Forward pass with activation collection
    activation_dict = {}
    attention_patterns = {}
    hooks = []
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._hook_fn(mod, inp, out, name, activation_dict, attention_patterns)
                )
            )

    # Classify if instruction is fact-based
    instruction = batch['instruction'][0] if isinstance(batch['instruction'], list) else batch['instruction']
    self.model.is_fact_based = self._classify_fact_based(instruction)

    # Forward pass with all data modalities
    outputs = self.model(
        batch['tokens']['bytes'],
        goal_context=None,
        brain_regions=batch.get('region_embeds'),
        rnn_states=None,
        sensory_data=batch.get('sensory_data'),
        tactile_data=batch.get('tactile_data'),
        audio_data=batch.get('audio_data'),
        image_data=batch.get('image_data')
    )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute hierarchical loss
    loss = self._compute_hierarchical_loss(outputs, batch)

    # Add specific loss for sensory/tactile training if needed
    if self.current_round == 'sensory_tactile_training':
        loss += self._compute_sensory_tactile_loss(outputs, batch)

    # Backward pass
    loss.backward()

    # Calculate per-layer gradient statistics
    layer_grads = {}
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
            grad = param.grad.detach()
            layer_grads[layer_name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': torch.norm(grad).item(),
                'sparsity': (grad == 0).float().mean().item()
            }

            # Gradient clipping
            if self.config.gradient_clip > 0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                grad_norm_after = torch.norm(param.grad.detach())

                metrics = {
                    'grad_norm_before_clip': grad_norm_before.item(),
                    'grad_norm_after_clip': grad_norm_after.item(),
                    'grad_clip_ratio': grad_norm_after.item() / (grad_norm_before.item() + 1e-8)
                }

                metrics.update({
                    f'grad_{k}': v for k, v in layer_grads.items()
                })

    # Step optimizer
    self.optimizer.step()
    self.scheduler.step()

    # Update episodic memory
    if self.config.use_episodic_memory:
        main_goal = self._extract_high_level_goal(batch['tokens']['bytes'])
        context = {
            'input': batch['tokens']['bytes'],
            'brain_regions': batch.get('region_embeds'),
            'step': self.global_step
        }
        subgoals = self.model._decompose_and_select_subgoals(main_goal, context)

        if 'reasoning_steps' in outputs and outputs['reasoning_steps']:
            final_output, reasoning_steps, final_reward = self.model._execute_with_chain_of_thought(subgoals, batch['tokens']['bytes'])

            self.model.store_experience(main_goal, context, subgoals, reasoning_steps, final_reward)
            self.model.goal_manager.update_tree(final_reward)
            retrieved_memories = self.model.retrieve_memories(main_goal, context, num_memories=self.config.num_retrieve_memories)
            self.model.sparse_update(retrieved_memories)
            self.model.update_memory_importance()

            if self.global_step % self.config.expansion_check_interval == 0:
                self.model.monitor_activations('subgoal_generator')
        else:
            print("Warning: No reasoning steps found in model output.")

    # Compute metrics
    metrics = self._compute_metrics(outputs, batch, loss.item())
    metrics.update({
        'grad_norm': grad_norm_before.item() if 'grad_norm_before_clip' in locals() else 0.0,
        'grad_mean': torch.mean(torch.stack([
            param.grad.mean()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_std': torch.std(torch.stack([
            param.grad.std()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_min': torch.min(torch.stack([
            param.grad.min()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item(),
        'grad_max': torch.max(torch.stack([
            param.grad.max()
            for param in self.model.parameters()
            if param.grad is not None
        ])).item()
    })

    # Add the hook function to the class
    def _hook_fn(self, module, input, output, name, activation_dict, attention_patterns):
        # Your existing hook logic
        pass

    # Add the compute_loss_from_memory method to the class
    def compute_loss_from_memory(self, memory):
      """
      Computes a loss based on a retrieved memory.
      This is a placeholder that you need to adapt based on the nature of your memories and your training objectives.
      """
      if memory['type'] == 'subgoal_generation':
          # Example: MSE loss between generated subgoal and memory subgoal
          generated_subgoal = self.subgoal_generator(torch.cat([memory['goal'], memory['context']], dim=-1))
          target_subgoal = memory['subgoal']
          loss = nn.MSELoss()(generated_subgoal, target_subgoal)

      elif memory['type'] == 'action_selection':
          # Example: Cross-entropy loss for action prediction
          # This requires adapting the model to output action probabilities
          # and having a way to compute the "correct" action from the memory
          logits = self.model(memory['context'])  # Assuming your model can take context and produce logits
          target_action = memory['action']  # This would need to be the index of the correct action
          loss = F.cross_entropy(logits, target_action)

      else:
          raise ValueError(f"Unknown memory type: {memory['type']}")

      return loss

    return metrics
        
        # Forward pass with activation collection
        activation_dict = {}
        attention_patterns = {}
        def hook_fn(module, input, output, name):
            # Store activation statistics and attention patterns
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                    activation_dict[name] = {
                        'hist': wandb.Histogram(act.cpu().numpy()),
                        'mean': act.mean().item(),
                        'std': act.std().item(),
                        'min': act.min().item(),
                        'max': act.max().item(),
                        'sparsity': (act == 0).float().mean().item()
                    }
                elif isinstance(output, tuple):
                    # For attention modules, output[0] is attention output, output[1] contains attention weights
                    act = output[0].detach()
                    activation_dict[name] = {
                        'hist': wandb.Histogram(act.cpu().numpy()),
                        'mean': act.mean().item(),
                        'std': act.std().item(),
                        'min': act.min().item(),
                        'max': act.max().item(),
                        'sparsity': (act == 0).float().mean().item()
                    }
                    
                    # Store attention patterns and head importance
                    if len(output) > 1 and isinstance(module, (nn.MultiheadAttention, CrossModalFusion)):
                        attn_weights = output[1].detach()  # [batch_size, num_heads, seq_len, seq_len]
                        
                        # Per-head attention patterns
                        for head in range(attn_weights.size(1)):
                            head_attention = attn_weights[:, head].mean(0).cpu()  # [seq_len, seq_len]
                            attention_patterns[f'{name}_head_{head}'] = wandb.Image(
                                wandb.plots.HeatMap(
                                    x_labels=[f'pos_{i}' for i in range(head_attention.shape[1])],
                                    y_labels=[f'pos_{i}' for i in range(head_attention.shape[0])],
                                    matrix_values=head_attention.numpy(),
                                    show_text=False
                                )
                            )
                            
                            # Compute head importance metrics
                            head_stats = {
                                # Attention entropy (lower means more focused)
                                'entropy': -(head_attention * torch.log(head_attention + 1e-10)).sum().item(),
                                
                                # Attention sparsity (higher means more selective)
                                'sparsity': (head_attention < 0.01).float().mean().item(),
                                
                                # Maximum attention weight (higher means stronger focus)
                                'max_attention': head_attention.max().item(),
                                
                                # Attention pattern stability (lower means more consistent)
                                'stability': head_attention.std().item(),
                                
                                # Position bias (how much it attends to nearby tokens)
                                'local_bias': self._compute_local_bias(head_attention)
                            }
                            
                            # Compute head attribution scores
                            with torch.enable_grad():
                                # Get output before head
                                pre_head = input[0].detach().requires_grad_()
                                
                                # Apply head attention
                                head_output = torch.matmul(
                                    head_attention,
                                    pre_head
                                )
                                
                                # Get gradients w.r.t predictions
                                if 'region_preds' in outputs:
                                    for region in outputs['region_ #Missing full variable name.
                                        grad = torch.autograd.grad(
                                            outputs['region_preds'][region].mean(),
                                            pre_head,
                                            retain_graph=True
                                        )[0]
                                        attribution = (grad * head_output).sum().item()
                                        head_stats[f'attribution_{region}'] = attribution
                                
                                if any(level in outputs for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']):
                                    for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
                                        if level in outputs:
                                            grad = torch.autograd.grad(
                                                outputs[level].mean(),
                                                pre_head,
                                                retain_graph=True
                                            )[0]
                                            attribution = (grad * head_output).sum().item()
                                            head_stats[f'attribution_{level}'] = attribution
                            
                            # Add head metrics
                            for metric_name, value in head_stats.items():
                                attention_patterns[f'{name}_head_{head}_{metric_name}'] = value
                                
                            # Analyze head redundancy
                            head_stats.update(self._analyze_head_redundancy(
                                module=module,
                                head_idx=head,
                                head_attention=head_attention,
                                input_tensor=input[0],
                                output_tensor=output[0],
                                batch=batch,
                                outputs=outputs
                            ))
                            
                            # Track head importance and redundancy
                            for metric_name, value in head_stats.items():
                                if metric_name.startswith('attribution_'):
                                    task = metric_name.split('_')[1]
                                    if task not in attention_patterns:
                                        attention_patterns[f'top_heads_{task}'] = []
                                    attention_patterns[f'top_heads_{task}'].append((f'{name}_head_{head}', value))
                                    
                                    # Track pruning candidates
                                    if head_stats.get('redundancy_score', 0) > 0.8:  # High redundancy threshold
                                        if 'pruning_candidates' not in attention_patterns:
                                            attention_patterns['pruning_candidates'] = []
                                        attention_patterns['pruning_candidates'].append({
                                            'head': f'{name}_head_{head}',
                                            'redundancy_score': head_stats['redundancy_score'],
                                            'attribution_score': value,
                                            'impact_score': head_stats.get('impact_score', 0),
                                            'similar_heads': head_stats.get('similar_heads', [])
                                        })
                                
                        # Overall attention pattern (averaged across heads)
                        avg_attention = attn_weights.mean(dim=(0, 1)).cpu()  # [seq_len, seq_len]
                        attention_patterns[f'{name}_overall'] = wandb.Image(
                            wandb.plots.HeatMap(
                                x_labels=[f'pos_{i}' for i in range(avg_attention.shape[1])],
                                y_labels=[f'pos_{i}' for i in range(avg_attention.shape[0])],
                                matrix_values=avg_attention.numpy(),
                                show_text=False
                            )
                        )

        # Register hooks for all modules
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                hooks.append(
                    module.register_forward_hook(
                        lambda mod, inp, out, name=name: hook_fn(mod, inp, out, name)
                    )
                )

        # Classify if instruction is fact-based
        instruction = batch['instruction'][0] if isinstance(batch['instruction'], list) else batch['instruction']
        self.model.is_fact_based = self._classify_fact_based(instruction)
        
        # Extract the main goal from the input batch
        main_goal = self._extract_high_level_goal(batch['tokens']['bytes'])

        # Decompose the main goal into subgoals and select relevant ones
        context = {
            'input': batch['tokens']['bytes'],  # Using byte-level tokens as context
            'brain_regions': batch.get('region_embeds'),  # Include brain region data if available
            'step': self.global_step
        }
        
        subgoals = self._decompose_and_select_subgoals(main_goal, context)
        
        # Execute subgoals with Chain-of-Thought and get the final output and reasoning steps
        final_output, reasoning_steps, final_reward = self._execute_with_chain_of_thought(subgoals, batch['tokens']['bytes'])
        
        # Update the goal tree based on the final reward
        self.goal_manager.update_tree(final_reward)
        
        # Use the final output for further processing or loss computation
        outputs = self.model(final_output, brain_regions=batch.get('region_embeds'), rnn_states=None)

        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute hierarchical loss
        loss = self._compute_hierarchical_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Calculate per-layer gradient statistics
        layer_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Get layer name (remove module prefixes and parameter suffixes)
                layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
                
                # Calculate statistics for this layer
                grad = param.grad.detach()
                layer_grads[layer_name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'norm': torch.norm(grad).item(),
                    'sparsity': (grad == 0).float().mean().item()
                }
            
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    grad_norm_after = torch.norm(param.grad.detach())
                    
                    metrics = {
                        'grad_norm_before_clip': grad_norm_before.item(),
                        'grad_norm_after_clip': grad_norm_after.item(),
                        'grad_clip_ratio': grad_norm_after.item() / (grad_norm_before.item() + 1e-8)
                    }
                    
                    metrics.update({
                        f'grad_{k}': v for k, v in layer_grads.items()
                    })
            
        # Step optimizer
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, batch, loss.item())
        metrics.update({
            'grad_norm': torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            ).item(),
            'grad_mean': torch.mean(torch.stack([
                param.grad.mean() 
                for param in self.model.parameters()
                if param.grad is not None
            ])).item(),
            'grad_std': torch.std(torch.stack([
                param.grad.std()
                for param in self.model.parameters() 
                if param.grad is not None
            ])).item(),
            'grad_min': torch.min(torch.stack([
                param.grad.min()
                for param in self.model.parameters()
                if param.grad is not None
            ])).item(),
            'grad_max': torch.max(torch.stack([
                param.grad.max()
        loss = loss + reg_loss
        
        # Return metrics with safe handling of None values
        metrics = {}
        
        # Add loss metrics
        metrics['loss'] = loss.item()
        
        # Add gradient metrics with safe handling of None values
        grad_metrics = {
            'grad_norm': grad_norm,
            'grad_norm_before_clip': grad_norm_before,
            'grad_norm_after_clip': grad_norm_after,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_min': grad_min,
            'grad_max': grad_max,
            'grad_clip_ratio': grad_clip_ratio
        }

      # Optionally, update the episodic memory
    if self.config.use_episodic_memory:
        # Example of storing an experience (adapt to your specific scenario)
        self.model.store_experience(
            goal=goal_embedding,
            context=context_embedding,
            subgoals=subgoal_embeddings,  # Assuming these are available
            actions=actions,  # Assuming you have a way to track actions
            rewards=rewards,
            reflection=reflection_text if reflection_needed else None
        )

        # Example of retrieving memories
        retrieved_memories = self.model.retrieve_memories(goal_embedding, context_embedding, num_memories=5)

        # Perform sparse updates based on retrieved memories
        self.model.sparse_update(retrieved_memories)

        # Update memory importance (e.g., with time decay)
        self.model.update_memory_importance()

        # Check for network expansion triggers
        if self.global_step % self.config.expansion_check_interval == 0:
            self.model.monitor_activations('subgoal_generator')

        
        # Safely convert tensor metrics to float values
        for key, value in grad_metrics.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
                else:
                    metrics[key] = float(value)
            else:
                metrics[key] = 0.0
                
        return metrics

    def _compute_hierarchical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute loss across hierarchical levels"""
        total_loss = 0
        
        # Loss for each hierarchical level
        for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
            if level in outputs and level in batch['tokens']:
                pred = outputs[level]
                target = batch['tokens'][level]
                
                # Cross entropy loss
                level_loss = nn.CrossEntropyLoss()(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1)
                )
                
                total_loss = total_loss + level_loss
        
        # Loss for brain region predictions
        if 'region_preds' in outputs and 'region_embeds' in batch:
            region_loss = nn.MSELoss()(
                outputs['region_preds'],
                batch['region_embeds']
            )
            total_loss = total_loss + region_loss
            
        return total_loss
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]],
        loss: float
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        with torch.no_grad():
            # Compute per-level accuracies
            level_metrics = {}
            total_correct = 0
            total_pred = 0
            
            for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
                if level in outputs and level in batch['tokens']:
                    pred = outputs[level]
                    target = batch['tokens'][level]
                    
                    # Get predictions
                    pred_ids = pred.argmax(dim=-1)
                    
                    # Count correct predictions
                    correct = (pred_ids == target).sum().item()
                    total = target.numel()
                    
                    # Store level-specific accuracy
                    level_metrics[f'{level}_accuracy'] = correct / total if total > 0 else 0
                    
                    # Accumulate for hierarchical accuracy
                    total_correct += correct
                    total_pred += total
            
            # Add level-specific metrics
            metrics.update(level_metrics)
            
            # Overall hierarchical accuracy
            metrics['hierarchical_accuracy'] = total_correct / total_pred if total_pred > 0 else 0
            
            # Compute accuracy for entropy patches
            if 'entropy_patches' in outputs and 'entropy_patches' in batch['tokens']:
                pred = outputs['entropy_patches']
                target = batch['tokens']['entropy_patches']
                
                # Get predictions
                pred_ids = pred.argmax(dim=-1)
                
                # Count correct predictions
                correct = (pred_ids == target).sum().item()
                total = target.numel()
                
                metrics['patch_accuracy'] = correct / total if total > 0 else 0
            
            # Compute brain region prediction accuracy
            if 'region_preds' in outputs and 'region_embeds' in batch:
                region_preds = outputs['region_preds']
                region_targets = batch['region_embeds']
                
                # Compute metrics for each region
                region_mse = {}
                region_accuracy = {}
                region_correlation = {}
                
                for region in ['visual', 'language', 'memory', 'attention', 'executive', 'semantic', 'integration', 'sensory_temporal', 'sensory_parietal']:
                    if region in region_preds and region in region_targets:
                        pred = region_preds[region]
                        target = region_targets[region]
                        
                        # MSE loss
                        mse = nn.MSELoss()(pred, target)
                        region_mse[f'{region}_mse'] = mse.item()
                        
                        # Cosine similarity as accuracy metric
                        cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
                        region_accuracy[f'{region}_accuracy'] = (cos_sim + 1) / 2  # Scale to [0,1]
                        
                        # Pearson correlation
                        pred_flat = pred.view(-1).detach().cpu().numpy()
                        target_flat = target.view(-1).detach().cpu().numpy()
                        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
                        region_correlation[f'{region}_correlation'] = correlation if not np.isnan(correlation) else 0.0
                
                # Add region-specific metrics
                metrics.update(region_mse)
                metrics.update(region_accuracy)
                
                # Overall brain region accuracy
                metrics['brain_region_accuracy'] = sum(region_accuracy.values()) / len(region_accuracy)
            
            # Compute overall accuracy (hierarchical + patches + brain regions)
            metrics['accuracy'] = (
                metrics.get('hierarchical_accuracy', 0) * 0.4 +
                metrics.get('patch_accuracy', 0) * 0.3 +
                metrics.get('brain_region_accuracy', 0) * 0.3
            )
            
            # Compute perplexity
            metrics['perplexity'] = torch.exp(torch.tensor(loss)).item()
            
            # Add loss
            metrics['loss'] = loss
        
        return metrics
    
    def _validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                # Move batch to device
                batch = {
                    k: {
                        k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
                        for k2, v2 in v.items()
                    }
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute factuality-aware loss
                if self.model.is_fact_based:
                    # For fact-based instructions, use factuality reward
                    factuality_scores = self.model.factuality_reward(outputs['hidden_states'])
                    supervision_weight = factuality_scores
                else:
                    # For non-fact-based instructions, use standard supervision
                    supervision_weight = torch.ones_like(outputs['logits'][:, :, 0])
                    
                # Apply factuality-aware weighting
                weighted_outputs = {
                    k: v * supervision_weight.unsqueeze(-1) if isinstance(v, torch.Tensor) and v.dim() == 3 else v
                    for k, v in outputs.items()
                }
                
                # Compute loss with factuality awareness
                loss = self._compute_hierarchical_loss(weighted_outputs, batch)
                
                # Compute metrics
                metrics = self._compute_metrics(outputs, batch, loss.item())
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_perplexity += metrics['perplexity']
        
        # Compute average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'perplexity': total_perplexity / num_batches
        }
        
        # Log metrics
        if self.config.use_wandb:
            log_metrics = {
                # Overall metrics
                'val/loss': avg_metrics['loss'],
                'val/hierarchical_accuracy': metrics['hierarchical_accuracy'],
                'val/patch_accuracy': metrics.get('patch_accuracy', 0),
                'val/brain_region_accuracy': metrics.get('brain_region_accuracy', 0),
                'val/combined_accuracy': metrics['accuracy'],
                'val/perplexity': avg_metrics['perplexity'],
                'val/step': self.global_step,
                
                # Per-level accuracies
                'val/bytes_accuracy': metrics.get('bytes_accuracy', 0),
                'val/chars_accuracy': metrics.get('chars_accuracy', 0),
                'val/words_accuracy': metrics.get('words_accuracy', 0),
                'val/sentences_accuracy': metrics.get('sentences_accuracy', 0),
                'val/paragraphs_accuracy': metrics.get('paragraphs_accuracy', 0),
                
                # Brain region accuracies
                'val/visual_accuracy': metrics.get('visual_accuracy', 0),
                'val/language_accuracy': metrics.get('language_accuracy', 0),
                'val/memory_accuracy': metrics.get('memory_accuracy', 0),
                'val/attention_accuracy': metrics.get('attention_accuracy', 0),
                'val/executive_accuracy': metrics.get('executive_accuracy', 0),
                'val/semantic_accuracy': metrics.get('semantic_accuracy', 0),
                'val/integration_accuracy': metrics.get('integration_accuracy', 0),
                'val/sensory_temporal_accuracy': metrics.get('sensory_temporal_accuracy', 0),
                'val/sensory_parietal_accuracy': metrics.get('sensory_parietal_accuracy', 0),
                
                # Brain region MSE
                'val/visual_mse': metrics.get('visual_mse', 0),
                'val/language_mse': metrics.get('language_mse', 0),
                'val/memory_mse': metrics.get('memory_mse', 0),
                'val/attention_mse': metrics.get('attention_mse', 0),
                'val/executive_mse': metrics.get('executive_mse', 0),
                'val/semantic_mse': metrics.get('semantic_mse', 0),
                'val/integration_mse': metrics.get('integration_mse', 0),
                'val/sensory_temporal_mse': metrics.get('sensory_temporal_mse', 0),
                'val/sensory_parietal_mse': metrics.get('sensory_parietal_mse', 0),
                
                # Brain region correlations
                'val/visual_correlation': metrics.get('visual_correlation', 0),
                'val/language_correlation': metrics.get('language_correlation', 0),
                'val/memory_correlation': metrics.get('memory_correlation', 0),
                'val/attention_correlation': metrics.get('attention_correlation', 0),
                'val/executive_correlation': metrics.get('executive_correlation', 0),
                'val/semantic_correlation': metrics.get('semantic_correlation', 0),
                'val/integration_correlation': metrics.get('integration_correlation', 0),
                'val/sensory_temporal_correlation': metrics.get('sensory_temporal_correlation', 0),
                'val/sensory_parietal_correlation': metrics.get('sensory_parietal_correlation', 0)
            }
            wandb.log(log_metrics)
        
        self.model.train()
        return avg_metrics
    
    def save_checkpoint(
        self,
        filename: str
    ) -> None:
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        save_path = self.config.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Path
    ) -> None:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        type=Path,
        help="Validation fMRI data directory containing NIfTI files"
    )
    
def main():
    parser = argparse.ArgumentParser(description="Train brain-aware BLT model with B-STAR")
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Training text data directory"
    )
    parser.add_argument(
        "--fmri-data",
        type=Path,
        help="Training fMRI data directory containing NIfTI files"
    )
    parser.add_argument(
        "--val-fmri-data",
        type=Path,
        help="Validation fMRI data directory containing NIfTI files"
    )

    parser.add_argument(
        "--fmri-window",
        type=int,
        default=200,
        help="Time window size for fMRI samples (from BrainLM paper)"
    )
    parser.add_argument(
        "--fmri-patch-size",
        type=int,
        default=20,
        help="Patch size for temporal signals (from BrainLM paper)"
    )
    parser.add_argument(
        "--n-parcels",
        type=int,
        default=424,
        help="Number of parcels for fMRI compression (from BrainLM paper)"
    )
    parser.add_argument(
        "--fmri-smooth-kernel",
        type=int,
        default=3,
        help="Size of Gaussian smoothing kernel for fMRI preprocessing"
    )
    parser.add_argument(
        "--fmri-min-activity",
        type=float,
        default=0.1,
        help="Minimum activity threshold for fMRI signals"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        help="Validation data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=1024,
        help="Size of B-STAR memory"
    )
    parser.add_argument(
        "--memory-topk",
        type=int,
        default=32,
        help="Number of top-k memories to retrieve"
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=1.0,
        help="Initial sampling temperature"
    )
    parser.add_argument(
        "--initial-exploration",
        type=float,
        default=0.1,
        help="Initial exploration rate"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Minimum confidence for memory updates"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum training steps"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model config with B-STAR and fMRI settings
        config = TrainingConfig(
            # B-STAR settings
            memory_size=args.memory_size,
            memory_topk=args.memory_topk,
            initial_temperature=args.initial_temperature,
            initial_exploration_rate=args.initial_exploration,
            min_confidence=args.min_confidence,
            
            # fMRI settings from BrainLM paper
            fmri_window=args.fmri_window,
            fmri_patch_size=args.fmri_patch_size,
            n_parcels=args.n_parcels,
            fmri_smooth_kernel=args.fmri_smooth_kernel,
            fmri_min_activity=args.fmri_min_activity,
            
            # Brain region mapping settings
            region_dim=256,  # Dimension for brain region embeddings
            num_region_heads=4,  # Number of attention heads for region mapping
            region_dropout=0.1,  # Dropout rate for region mapping
            anatomical_constraint_weight=0.5,  # Weight for anatomical constraints
            
            # Transformer settings
            d_model=512,  # Model dimension
            n_heads=8,  # Number of attention heads
            encoder_layers=4,  # Number of encoder layers
            decoder_layers=4,  # Number of decoder layers
            dropout=0.1,  # Dropout rate
            
            # Training settings
            batch_size=32,
            learning_rate=1e-4,
            warmup_steps=1000,
            max_steps=args.max_steps,
            gradient_clip=1.0,

            # Loss settings
            reconstruction_weight=1.0,  # Weight for fMRI reconstruction loss
            region_prediction_weight=0.5,  # Weight for brain region prediction loss
            anatomical_loss_weight=0.1  # Weight for anatomical constraint loss
        )

        # Create model
        model = ByteLatentTransformer(config)
        
        # Load datasets
        logger.info("Loading datasets...")
        text_data = load_text_data(args.train_data)
        fmri_data = load_fmri_data(args.fmri_data) if args.fmri_data else None
        
        logger.info(f"Loaded {len(text_data)} text samples")
        if fmri_data is not None:
            logger.info(f"Loaded fMRI data with shape {fmri_data.shape}")
        
        train_dataset = MultimodalBrainAwareDataset(
            text_data=text_data,
            fmri_data=fmri_data,
            config=config,
            augment_prob=0.5,  # Enable data augmentation
            mask_prob=0.15,    # Enable masking for MLM-style training
            fmri_window=200,   # From BrainLM paper
            fmri_patch_size=20,  # From BrainLM paper
            n_parcels=424      # From BrainLM paper
        )
        
        val_dataset = None
        if args.val_data:
            val_text_data = load_text_data(args.val_data)
            val_fmri_data = load_fmri_data(args.val_fmri_data) if args.val_fmri_data else None
            
            val_dataset = MultimodalBrainAwareDataset(
                text_data=val_text_data,
                fmri_data=val_fmri_data,
                config=config,
                augment_prob=0.0,  # No augmentation for validation
                mask_prob=0.0,     # No masking for validation
                fmri_window=200,   # From BrainLM paper
                fmri_patch_size=20,  # From BrainLM paper
                n_parcels=424      # From BrainLM paper
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Create B-STAR trainer
        trainer = BrainAwareBLTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )


   
        #Episodic Memory Nuerogensis and Clustering occasionally activated during training. 
        if self.global_step % self.config.cluster_maintenance_interval == 0:
            self.model.episodic_memory.perform_cluster_maintenance()
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train model with B-STAR
        logger.info("Starting B-STAR training...")
        trainer.train()

        # After each epoch or at certain intervals:
        
       # Save final model
      final_path_pt = args.checkpoint_dir / "final_model.pt"
      final_path_safetensors = args.checkpoint_dir / "final_model.safetensors"

      logger.info(f"Saving final model (pytorch) to {final_path_pt}")
      trainer.save_model(final_path_pt)  # Use save_model for saving in .pt format

      logger.info(f"Saving final model (safetensors) to {final_path_safetensors}")
      trainer.save_model(final_path_safetensors, safe_serialization=True) # Use save_model with safe_serialization for .safetensors
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def load_text_data(data_dir: Path) -> List[str]:
    """Load text data from directory"""
    texts = []
    for file in data_dir.glob("*.txt"):
        with open(file, "r") as f:
            texts.extend(f.read().split("\n\n"))
    return texts

def load_fmri_data(data_dir: Path, smooth_kernel: int = 3, min_activity: float = 0.1) -> Optional[torch.Tensor]:
    """Load and preprocess fMRI data from NIfTI files according to BrainLM paper"""
    try:
        import nibabel as nib
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError("nibabel and scipy are required for fMRI data loading. Install with: pip install nibabel scipy")
        
    # Load NIfTI files
    fmri_files = list(data_dir.glob("*.nii.gz"))
    if not fmri_files:
        return None
        
    fmri_data = []
    for file in fmri_files:
        # Load NIfTI using nibabel
        img = nib.load(file)
        data = img.get_fdata()
        
        # Apply spatial smoothing with Gaussian kernel
        for t in range(data.shape[-1]):  # Smooth each time point
            data[..., t] = gaussian_filter(data[..., t], sigma=smooth_kernel)
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        # Reshape to (time, x, y, z)
        if len(data.shape) == 4:
            data = data.permute(3, 0, 1, 2)
            
        # Normalize each voxel time series
        data = (data - data.mean(dim=0, keepdim=True)) / (data.std(dim=0, keepdim=True) + 1e-8)
        
        # Apply activity threshold
        data = data * (torch.abs(data) > min_activity).float()
        
        fmri_data.append(data)
    
    # Stack all sessions
    fmri_data = torch.cat(fmri_data, dim=0)
    
    return fmri_data

if __name__ == "__main__":
    main()

''' #The below is needed for training the self-configuration (SVF) of the model depending on the task that the user asks the model to perform from the visual to the language_understanding tasks. This will need to be
    #properly added to the above program to work properly with the rest of the program. 
    # Model and tokenizer initialization
    model_name = "this model being trained"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tasks and SVF initialization
    tasks = ["math", "language_understanding", "code", "visual", "smell", "tactile", "motor"]  # Define your tasks
    svf = SVF(model, tasks, rank=32, alpha=0.5, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    data_filepath = "your_task_dataset.csv"  # Replace with your dataset file
    dataset = TaskDataset(data_filepath, tokenizer)

    # Train z-vectors
    train_z_vectors(
        model,
        svf,
        dataset,
        tokenizer,
        num_episodes=100,
        batch_size=4,
        learning_rate=1e-4,
        gamma=0.99,
        clip_ratio=0.2
    )
'''
                  
