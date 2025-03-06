import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
# Import Grok optimizers
from GrokOptimizers import OrthoAdamW, OrthoGrad, OrthoSGD, StableCrossEntropyLoss, StableMax

class CrossAttention(nn.Module):
    """
    Cross-attention module for integrating dynamic byte patches in empathy calculations.
    This enables the model to attend to specific state representations across perspectives.
    Now with StableMax for improved numerical stability.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.stablemax = StableMax()  # Use StableMax for more numerically stable attention
        
        # For TOVA integration
        self.use_kv_cache = False
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        
    def enable_kv_caching(self):
        """Enable KV caching for TOVA compression"""
        self.use_kv_cache = True
        
    def reset_cache(self):
        """Reset KV caches"""
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        
    def forward(self, query, key_value):
        """
        Apply cross-attention between query and key_value tensors.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, embed_dim]
            key_value: Key-value tensor of shape [batch_size, seq_len_kv, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len_q, embed_dim]
        """
        batch_size = query.shape[0]
        
        # Project queries, keys, and values
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Store key and value for KV caching if enabled
        if self.use_kv_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                # Append to cache
                self.k_cache = torch.cat([self.k_cache, k], dim=2)  # Concat along seq_len dim
                self.v_cache = torch.cat([self.v_cache, v], dim=2)  # Concat along seq_len dim
                
                # Use cached keys and values
                k = self.k_cache
                v = self.v_cache
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Use StableMax instead of softmax for greater numerical stability
        attention_weights = self.stablemax(scores)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for TOVA if KV caching is enabled
        if self.use_kv_cache:
            self.attention_weights = attention_weights
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Apply output projection
        output = self.output_proj(context)
        
        return output

class EntropyPredictor(nn.Module):
    """
    Predicts entropy for dynamic patch boundaries based on state and action sequences.
    Used to determine where to create patch boundaries for efficient processing.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.entropy_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Predict entropy for each position in the sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Entropy predictions of shape [batch_size, seq_len, 1]
        """
        encoded = self.encoder(x)
        entropy = self.entropy_head(encoded)
        
        # Apply sigmoid to get entropy values between 0 and 1
        entropy = torch.sigmoid(entropy)
        
        return entropy

class MirrorNeuronEmpathyReward(nn.Module):
    """
    Implementation of the Mirror Neuron Empathy Reward component with dynamic byte patches,
    cross-attention, and entropy processing for the COCONUT Latent class model.
    
    This module calculates empathy rewards based on mirror neuron theory, which allows the LLM
    to simulate and respond to the emotional states of others.
    
    The core formula implemented is:
    R_{emp}^{mirror}(s, a) = w_{mirror} * (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
    
    Now integrated with dynamic patching and cross-attention for improved processing.
    """
    def __init__(
        self,
        embedding_dim: int,
        mirror_weight: float = 0.7,
        num_perspectives: int = 4,
        entropy_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_tova: bool = True,
        num_heads: int = 4,
        cache_max_size: int = 512,
    ):
        """
        Initialize the Mirror Neuron Empathy Reward component with support for dynamic byte patching.
        
        Args:
            embedding_dim: Dimension of state and action embeddings
            mirror_weight: Weight for the mirror neuron empathy component (w_{mirror})
            num_perspectives: Number of different Q-functions to use (N)
            entropy_threshold: Threshold for determining patch boundaries based on entropy
            device: Device to run computations on
            use_tova: Whether to use TOVA compression for KV caches
            num_heads: Number of attention heads in cross-attention
            cache_max_size: Maximum size of KV cache for TOVA compression
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mirror_weight = mirror_weight
        self.num_perspectives = num_perspectives
        self.entropy_threshold = entropy_threshold
        self.device = device
        self.use_tova = use_tova
        self.cache_max_size = cache_max_size
        
        # Create cross-attention module for perspective integration
        self.cross_attention = CrossAttention(embedding_dim, num_heads=num_heads)
        
        # Create entropy predictor for dynamic patching
        self.entropy_predictor = EntropyPredictor(embedding_dim * 2)
        
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
        
        # If TOVA is enabled, initialize KV caching
        if use_tova:
            self.cross_attention.enable_kv_caching()
    
    def calculate_shannon_entropy(self, probs):
        """
        Calculate Shannon entropy from a probability distribution.
        
        Args:
            probs: Probability tensor of shape [batch_size, vocab_size]
            
        Returns:
            Entropy value tensor of shape [batch_size]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-9
        log_probs = torch.log2(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def dynamic_patching(self, states, actions):
        """
        Apply dynamic patching based on entropy prediction.
        
        Args:
            states: Tensor of states [batch_size, seq_len, embedding_dim]
            actions: Tensor of actions [batch_size, seq_len, embedding_dim]
            
        Returns:
            Patched states and actions with patch boundary indices
        """
        batch_size, seq_len, _ = states.shape
        
        # Concatenate states and actions for entropy prediction
        state_action_pairs = torch.cat([states, actions], dim=2)
        
        # Predict entropy
        entropies = self.entropy_predictor(state_action_pairs)
        
        # Find patch boundaries where entropy exceeds threshold
        patch_boundaries = (entropies > self.entropy_threshold).squeeze(-1)
        
        # Create patches
        patched_states = []
        patched_actions = []
        patch_indices = []
        
        for b in range(batch_size):
            # Get boundaries for this batch item
            boundaries = torch.nonzero(patch_boundaries[b]).squeeze(-1)
            
            # Add sequence end as final boundary if not already included
            if boundaries.shape[0] == 0 or boundaries[-1] != seq_len - 1:
                boundaries = torch.cat([boundaries, torch.tensor([seq_len - 1], device=boundaries.device)])
            
            # Initialize start index
            start_idx = 0
            
            # Process each patch
            for end_idx in boundaries:
                # Extract patch
                state_patch = states[b, start_idx:end_idx+1]
                action_patch = actions[b, start_idx:end_idx+1]
                
                # Average pooling for the patch representation
                pooled_state = state_patch.mean(dim=0, keepdim=True)
                pooled_action = action_patch.mean(dim=0, keepdim=True)
                
                # Add to patched tensors
                patched_states.append(pooled_state)
                patched_actions.append(pooled_action)
                patch_indices.append((b, start_idx, end_idx))
                
                # Update start index for next patch
                start_idx = end_idx + 1
        
        # Stack patches if any exist
        if patched_states:
            patched_states = torch.cat(patched_states, dim=0)
            patched_actions = torch.cat(patched_actions, dim=0)
        else:
            # Fallback if no patches were created
            patched_states = states.view(-1, self.embedding_dim)
            patched_actions = actions.view(-1, self.embedding_dim)
            patch_indices = [(b, 0, seq_len-1) for b in range(batch_size)]
        
        return patched_states, patched_actions, patch_indices
    
    def apply_tova_compression(self):
        """Apply TOVA compression to cross-attention KV cache if enabled"""
        if not self.use_tova or not hasattr(self.cross_attention, 'attention_weights'):
            return
            
        if (self.cross_attention.k_cache is not None and
            self.cross_attention.v_cache is not None and
            self.cross_attention.attention_weights is not None):
            
            # Apply TOVA compression (in a real implementation, this would use the TOVACompression module)
            # Here we're simulating compression by keeping tokens with highest attention weights
            attention_sum = self.cross_attention.attention_weights.mean(dim=(0, 1))  # Average across batch and heads
            
            # Keep only the tokens with highest attention weights up to cache_max_size
            if self.cross_attention.k_cache.size(2) > self.cache_max_size:
                _, top_indices = torch.topk(attention_sum, self.cache_max_size)
                self.cross_attention.k_cache = torch.index_select(self.cross_attention.k_cache, 2, top_indices)
                self.cross_attention.v_cache = torch.index_select(self.cross_attention.v_cache, 2, top_indices)
    
    def forward(
        self,
        self_state: torch.Tensor,
        other_state: torch.Tensor,
        action: torch.Tensor,
        no_action: Optional[torch.Tensor] = None,
        apply_dynamic_patching: bool = True
    ) -> torch.Tensor:
        """
        Calculate the mirror neuron empathy reward with dynamic patching and cross-attention.
        
        Args:
            self_state: Tensor representing the agent's own state [batch_size, seq_len, embedding_dim]
            other_state: Tensor representing the other's state [batch_size, seq_len, embedding_dim]
            action: Tensor representing the action taken [batch_size, seq_len, embedding_dim]
            no_action: Tensor representing the null action (inaction) [batch_size, seq_len, embedding_dim]
                If None, will use a zero tensor of appropriate size
            apply_dynamic_patching: Whether to apply dynamic patching based on entropy
        
        Returns:
            Mirror neuron empathy reward
        """
        batch_size = self_state.shape[0]
        
        # If no_action is not provided, use a zero tensor
        if no_action is None:
            no_action = torch.zeros_like(action)
        
        # Apply dynamic patching if enabled
        if apply_dynamic_patching:
            self_state_patched, action_patched, self_patches = self.dynamic_patching(self_state, action)
            other_state_patched, no_action_patched, other_patches = self.dynamic_patching(other_state, no_action)
        else:
            # Fallback to original tensors if no patching
            if self_state.dim() == 3:  # [batch_size, seq_len, embedding_dim]
                self_state_patched = self_state.reshape(-1, self.embedding_dim)
                action_patched = action.reshape(-1, self.embedding_dim)
                other_state_patched = other_state.reshape(-1, self.embedding_dim)
                no_action_patched = no_action.reshape(-1, self.embedding_dim)
            else:
                self_state_patched = self_state
                action_patched = action
                other_state_patched = other_state
                no_action_patched = no_action
        
        # Apply cross-attention for perspective enhancement
        # Use self state as query and other state as key-value
        enhanced_other_state = self.cross_attention(self_state_patched.unsqueeze(1),
                                                  other_state_patched.unsqueeze(1)).squeeze(1)
        
        # Apply TOVA compression if enabled
        if self.use_tova:
            self.apply_tova_compression()
        
        q_values_action = []
        q_values_no_action = []
        
        for q_net in self.q_networks:
            # Concatenate enhanced other state with action
            q_input_action = torch.cat([enhanced_other_state, action_patched], dim=1)
            q_value_action = q_net(q_input_action)
            q_values_action.append(q_value_action)
            
            # Concatenate enhanced other state with no action
            q_input_no_action = torch.cat([enhanced_other_state, no_action_patched], dim=1)
            q_value_no_action = q_net(q_input_no_action)
            q_values_no_action.append(q_value_no_action)
        
        # Stack Q-values and calculate average difference
        q_values_action = torch.stack(q_values_action, dim=1)  # [num_patches, num_perspectives, 1]
        q_values_no_action = torch.stack(q_values_no_action, dim=1)  # [num_patches, num_perspectives, 1]
        
        # Calculate empathy reward as the average change in Q-values
        # R_{emp}(s, a) = (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
        empathy_reward = (q_values_action - q_values_no_action).mean(dim=1)  # [num_patches, 1]
        
        # Aggregate patch rewards by batch
        if apply_dynamic_patching:
            batch_rewards = defaultdict(list)
            for (b, _, _), reward in zip(self_patches, empathy_reward):
                batch_rewards[b].append(reward)
            
            # Average rewards for each batch item
            aggregated_rewards = torch.stack([
                torch.stack(batch_rewards[b]).mean() if b in batch_rewards else torch.zeros(1, device=self.device)
                for b in range(batch_size)
            ])
        else:
            # Reshape rewards to match batch size if no patching was applied
            aggregated_rewards = empathy_reward.view(batch_size, -1).mean(dim=1)
        
        # Apply mirror weight
        # R_{emp}^{mirror}(s, a) = w_{mirror} * (1/N) \sum_{i=1}^{N} [Q_i(s^{others}, a) - Q_i(s^{others}, \emptyset)]
        mirror_empathy_reward = self.mirror_weight * aggregated_rewards
        
        return mirror_empathy_reward.unsqueeze(1)  # [batch_size, 1]


class NegativeEnvironmentalImpactAvoidance(nn.Module):
    """
    Implementation of the Side-Effect Penalty (Environmental Negative Avoidance) component
    with support for dynamic byte patches, cross-attention, and entropy processing.
    
    This module calculates penalties for actions that negatively impact the environment:
    R_{nse}(s, a) = (1/N) \sum_{i=1}^{N} \max(0, -(Q_i(s, a) - Q_i(s, \emptyset)))
    """
    def __init__(
        self,
        embedding_dim: int,
        num_perspectives: int = 4,
        entropy_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_tova: bool = True,
        num_heads: int = 4,
        cache_max_size: int = 512,
    ):
        """
        Initialize the Negative Environmental Impact Avoidance component with patching support.
        
        Args:
            embedding_dim: Dimension of state and action embeddings
            num_perspectives: Number of different Q-functions to use (N)
            entropy_threshold: Threshold for determining patch boundaries based on entropy
            device: Device to run computations on
            use_tova: Whether to use TOVA compression for KV caches
            num_heads: Number of attention heads in cross-attention
            cache_max_size: Maximum size of KV cache for TOVA compression
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_perspectives = num_perspectives
        self.entropy_threshold = entropy_threshold
        self.device = device
        self.use_tova = use_tova
        self.cache_max_size = cache_max_size
        
        # Create cross-attention module for state-action integration
        self.cross_attention = CrossAttention(embedding_dim, num_heads=num_heads)
        
        # Create entropy predictor for dynamic patching
        self.entropy_predictor = EntropyPredictor(embedding_dim * 2)
        
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
        
        # If TOVA is enabled, initialize KV caching
        if use_tova:
            self.cross_attention.enable_kv_caching()
    
    def dynamic_patching(self, states, actions):
        """
        Apply dynamic patching based on entropy prediction.
        
        Args:
            states: Tensor of states [batch_size, seq_len, embedding_dim]
            actions: Tensor of actions [batch_size, seq_len, embedding_dim]
            
        Returns:
            Patched states and actions with patch boundary indices
        """
        batch_size, seq_len, _ = states.shape
        
        # Concatenate states and actions for entropy prediction
        state_action_pairs = torch.cat([states, actions], dim=2)
        
        # Predict entropy
        entropies = self.entropy_predictor(state_action_pairs)
        
        # Find patch boundaries where entropy exceeds threshold
        patch_boundaries = (entropies > self.entropy_threshold).squeeze(-1)
        
        # Create patches
        patched_states = []
        patched_actions = []
        patch_indices = []
        
        for b in range(batch_size):
            # Get boundaries for this batch item
            boundaries = torch.nonzero(patch_boundaries[b]).squeeze(-1)
            
            # Add sequence end as final boundary if not already included
            if boundaries.shape[0] == 0 or boundaries[-1] != seq_len - 1:
                boundaries = torch.cat([boundaries, torch.tensor([seq_len - 1], device=boundaries.device)])
            
            # Initialize start index
            start_idx = 0
            
            # Process each patch
            for end_idx in boundaries:
                # Extract patch
                state_patch = states[b, start_idx:end_idx+1]
                action_patch = actions[b, start_idx:end_idx+1]
                
                # Average pooling for the patch representation
                pooled_state = state_patch.mean(dim=0, keepdim=True)
                pooled_action = action_patch.mean(dim=0, keepdim=True)
                
                # Add to patched tensors
                patched_states.append(pooled_state)
                patched_actions.append(pooled_action)
                patch_indices.append((b, start_idx, end_idx))
                
                # Update start index for next patch
                start_idx = end_idx + 1
        
        # Stack patches if any exist
        if patched_states:
            patched_states = torch.cat(patched_states, dim=0)
            patched_actions = torch.cat(patched_actions, dim=0)
        else:
            # Fallback if no patches were created
            patched_states = states.view(-1, self.embedding_dim)
            patched_actions = actions.view(-1, self.embedding_dim)
            patch_indices = [(b, 0, seq_len-1) for b in range(batch_size)]
        
        return patched_states, patched_actions, patch_indices
    
    def apply_tova_compression(self):
        """Apply TOVA compression to cross-attention KV cache if enabled"""
        if not self.use_tova or not hasattr(self.cross_attention, 'attention_weights'):
            return
            
        if (self.cross_attention.k_cache is not None and
            self.cross_attention.v_cache is not None and
            self.cross_attention.attention_weights is not None):
            
            # Apply TOVA compression (in a real implementation, this would use the TOVACompression module)
            # Here we're simulating compression by keeping tokens with highest attention weights
            attention_sum = self.cross_attention.attention_weights.mean(dim=(0, 1))  # Average across batch and heads
            
            # Keep only the tokens with highest attention weights up to cache_max_size
            if self.cross_attention.k_cache.size(2) > self.cache_max_size:
                _, top_indices = torch.topk(attention_sum, self.cache_max_size)
                self.cross_attention.k_cache = torch.index_select(self.cross_attention.k_cache, 2, top_indices)
                self.cross_attention.v_cache = torch.index_select(self.cross_attention.v_cache, 2, top_indices)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        no_action: Optional[torch.Tensor] = None,
        apply_dynamic_patching: bool = True
    ) -> torch.Tensor:
        """
        Calculate the environmental negative impact avoidance penalty with patching support.
        
        Args:
            state: Tensor representing the environment state [batch_size, seq_len, embedding_dim]
            action: Tensor representing the action taken [batch_size, seq_len, embedding_dim]
            no_action: Tensor representing the null action (inaction) [batch_size, seq_len, embedding_dim]
                If None, will use a zero tensor of appropriate size
            apply_dynamic_patching: Whether to apply dynamic patching based on entropy
        
        Returns:
            Environmental negative impact penalty [batch_size, 1]
        """
        batch_size = state.shape[0]
        
        # If no_action is not provided, use a zero tensor
        if no_action is None:
            no_action = torch.zeros_like(action)
        
        # Apply dynamic patching if enabled
        if apply_dynamic_patching and state.dim() == 3:  # Need sequence dimension for patching
            state_patched, action_patched, state_patches = self.dynamic_patching(state, action)
            state_patched2, no_action_patched, no_action_patches = self.dynamic_patching(state, no_action)
        else:
            # Fallback to original tensors if no patching
            if state.dim() == 3:  # [batch_size, seq_len, embedding_dim]
                state_patched = state.reshape(-1, self.embedding_dim)
                action_patched = action.reshape(-1, self.embedding_dim)
                state_patched2 = state.reshape(-1, self.embedding_dim)
                no_action_patched = no_action.reshape(-1, self.embedding_dim)
                state_patches = [(b, 0, state.shape[1]-1) for b in range(batch_size)]
            else:
                state_patched = state
                action_patched = action
                state_patched2 = state
                no_action_patched = no_action
                state_patches = [(b, 0, 0) for b in range(batch_size)]
        
        # Apply cross-attention for state-action integration
        enhanced_state_action = self.cross_attention(state_patched.unsqueeze(1),
                                                   action_patched.unsqueeze(1)).squeeze(1)
        
        enhanced_state_no_action = self.cross_attention(state_patched2.unsqueeze(1),
                                                      no_action_patched.unsqueeze(1)).squeeze(1)
        
        # Apply TOVA compression if enabled
        if self.use_tova:
            self.apply_tova_compression()
        
        q_values_action = []
        q_values_no_action = []
        
        for q_net in self.env_q_networks:
            # Concatenate enhanced state with action
            q_input_action = torch.cat([enhanced_state_action, action_patched], dim=1)
            q_value_action = q_net(q_input_action)
            q_values_action.append(q_value_action)
            
            # Concatenate enhanced state with no action
            q_input_no_action = torch.cat([enhanced_state_no_action, no_action_patched], dim=1)
            q_value_no_action = q_net(q_input_no_action)
            q_values_no_action.append(q_value_no_action)
        
        # Stack Q-values and calculate average difference
        q_values_action = torch.stack(q_values_action, dim=1)  # [num_patches, num_perspectives, 1]
        q_values_no_action = torch.stack(q_values_no_action, dim=1)  # [num_patches, num_perspectives, 1]
        
        # Calculate the environmental penalty using max(0, -difference)
        # R_{nse}(s, a) = (1/N) \sum_{i=1}^{N} \max(0, -(Q_i(s, a) - Q_i(s, \emptyset)))
        q_diff = q_values_action - q_values_no_action
        patch_penalties = torch.max(torch.zeros_like(q_diff), -q_diff).mean(dim=1)  # [num_patches, 1]
        
        # Aggregate patch penalties by batch
        if apply_dynamic_patching and state.dim() == 3:
            batch_penalties = defaultdict(list)
            for (b, _, _), penalty in zip(state_patches, patch_penalties):
                batch_penalties[b].append(penalty)
            
            # Average penalties for each batch item
            aggregated_penalties = torch.stack([
                torch.stack(batch_penalties[b]).mean() if b in batch_penalties else torch.zeros(1, device=self.device)
                for b in range(batch_size)
            ])
        else:
            # Reshape penalties to match batch size if no patching was applied
            aggregated_penalties = patch_penalties.view(batch_size, -1).mean(dim=1)
        
        return aggregated_penalties.unsqueeze(1)  # [batch_size, 1]


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
    Implementation of the full moral reward calculation with COCONUT Latent model support.
    
    This class combines all reward components to calculate the total moral reward:
    R_{moral}(t) = R_{self-task}(t_{end}) + DA_{in-emp}(t) + R_{emp}^{mirror}(s, a) +
                     R_{nse}(s, a) + R_{penalty}(t) + R_{perspective_taking}(s, a) +
                     R_{episodic_memory}(s, a) + R_{altruism_longterm}(s, a)
                     
    Now with dynamic byte patches, cross-attention, and entropy processing for the COCONUT model.
    
    The long-term altruism reward component leverages episodic memory and human feedback
    to reinforce altruistic behaviors over longer timescales.
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
        entropy_threshold: float = 0.8,
        use_tova: bool = True,
        num_heads: int = 4,
        cache_max_size: int = 512,
        episodic_memory = None,
        positive_feedback_value: float = 1.0,
        negative_feedback_penalty: float = -1.5,
        feedback_reward_scale: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Full Moral Reward Calculator with COCONUT model support.
        
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
            entropy_threshold: Threshold for determining patch boundaries based on entropy
            use_tova: Whether to use TOVA compression for KV caches
            num_heads: Number of attention heads in cross-attention
            cache_max_size: Maximum size of KV cache for TOVA compression
            episodic_memory: Instance of EpisodicMemory for long-term altruism reward
            positive_feedback_value: Value for positive feedback (for long-term altruism)
            negative_feedback_penalty: Penalty for negative feedback (for long-term altruism)
            feedback_reward_scale: Scaling factor for long-term altruism feedback
            device: Device to run computations on
        """
        self.embedding_dim = embedding_dim
        self.self_task_target = self_task_target
        self.device = device
        self.use_tova = use_tova
        self.episodic_memory = episodic_memory
        
        # Initialize reward components with COCONUT model support
        self.mirror_empathy = MirrorNeuronEmpathyReward(
            embedding_dim=embedding_dim,
            mirror_weight=mirror_weight,
            num_perspectives=num_perspectives,
            entropy_threshold=entropy_threshold,
            device=device,
            use_tova=use_tova,
            num_heads=num_heads,
            cache_max_size=cache_max_size
        )
        
        self.env_penalty = NegativeEnvironmentalImpactAvoidance(
            embedding_dim=embedding_dim,
            num_perspectives=num_perspectives,
            entropy_threshold=entropy_threshold,
            device=device,
            use_tova=use_tova,
            num_heads=num_heads,
            cache_max_size=cache_max_size
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
        
        # Initialize long-term altruism reward component if episodic memory is provided
        self.long_term_altruism = None
        if episodic_memory is not None:
            from LongTermAltruismReward import LongTermAltruismReward
            self.long_term_altruism = LongTermAltruismReward(
                episodic_memory=episodic_memory,
                positive_feedback_value=positive_feedback_value,
                negative_feedback_penalty=negative_feedback_penalty,
                feedback_reward_scale=feedback_reward_scale,
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
        human_agent_id: Optional[str] = None,
        time_step: float = 0.02,
        apply_dynamic_patching: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the full moral reward with COCONUT model support.
        
        Args:
            self_state: Tensor representing the agent's own state [batch_size, seq_len, embedding_dim]
            other_state: Tensor representing the other's state [batch_size, seq_len, embedding_dim]
            action: Tensor representing the action taken [batch_size, seq_len, embedding_dim]
            no_action: Tensor representing the null action (inaction) [batch_size, seq_len, embedding_dim]
            empathy_signal: Tensor representing the current empathy signal S(t)
            emotion_value: Tensor representing the current emotion value
            is_end_of_episode: Boolean indicating if this is the end of the episode
            perspective_taking_reward: Additional reward for perspective taking
            episodic_memory_reward: Additional reward for episodic memory
            human_agent_id: Optional ID of the human agent for long-term altruism calculation
            time_step: Time step duration in seconds
            apply_dynamic_patching: Whether to apply dynamic patching based on entropy
        
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
            "altruism_longterm": torch.tensor([0.0], device=self.device),
        }
        
        # Calculate mirror neuron empathy reward with dynamic patching
        if self_state is not None and other_state is not None and action is not None:
            rewards["mirror_empathy"] = self.mirror_empathy(
                self_state=self_state,
                other_state=other_state,
                action=action,
                no_action=no_action,
                apply_dynamic_patching=apply_dynamic_patching
            )
        
        # Calculate environmental negative impact penalty with dynamic patching
        if self_state is not None and action is not None:
            rewards["env_penalty"] = self.env_penalty(
                state=self_state,
                action=action,
                no_action=no_action,
                apply_dynamic_patching=apply_dynamic_patching
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
        
        # Calculate long-term altruism reward if available
        if self.long_term_altruism is not None and self_state is not None and action is not None:
            rewards["altruism_longterm"] = self.long_term_altruism(
                state=self_state,
                action=action,
                human_agent_id=human_agent_id
            )
        
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
        use_stable_loss: bool = True,
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
        self.use_stable_loss = use_stable_loss
        
        # Initialize loss function - use StableCrossEntropyLoss for better stability if requested
        if use_stable_loss:
            self.loss_fn = get_stable_loss_function()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            
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
        
        # Create optimizer for encoders - use OrthoAdamW from GrokOptimizers for better stability
        self.optimizer = OrthoAdamW(
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
        
        # Encode question as state (self perspective)
        question_embedding = self.encode_text(question)
        self_state_embedding = self.state_encoder(question_embedding)
        
        # Create other's perspective state with slight variation for mirror neuron empathy
        other_state_embedding = self_state_embedding.clone()
        # Add small random noise to create a different perspective
        other_state_embedding = other_state_embedding + torch.randn_like(other_state_embedding) * 0.1
        
        # For each choice, compute comprehensive moral rewards using all components
        choice_rewards = []
        choice_reward_components = []
        
        for choice in choices:
            # Encode choice as action
            choice_embedding = self.encode_text(choice)
            action_embedding = self.action_encoder(choice_embedding)
            
            # Create null action (no action alternative)
            no_action = torch.zeros_like(action_embedding)
            
            # Generate empathy signal based on whether this is the correct answer
            # Higher empathy signal for correct choices (ethical decisions)
            empathy_signal = torch.tensor([0.8 if choice == correct_answer else 0.2], device=self.device)
            
            # Generate emotion value based on whether this is the correct answer
            # Positive emotion for correct choices, negative for incorrect
            emotion_value = torch.tensor([0.5 if choice == correct_answer else -0.3], device=self.device)
            
            # Calculate comprehensive moral reward using all components
            rewards = self.moral_reward_calculator.calculate_reward(
                self_state=self_state_embedding,
                other_state=other_state_embedding,
                action=action_embedding,
                no_action=no_action,
                empathy_signal=empathy_signal,
                emotion_value=emotion_value,
                # Enable dynamic patching for complex moral scenarios
                apply_dynamic_patching=True,
                # Simulate an episode end to get self-task rewards for complete moral evaluation
                is_end_of_episode=(choice == correct_answer),
                # Add perspective taking and episodic memory rewards if this is a correct choice
                perspective_taking_reward=0.5 if choice == correct_answer else 0.0,
                episodic_memory_reward=0.3 if choice == correct_answer else 0.0
            )
            
            # Store the total reward and components for analysis
            choice_rewards.append(rewards["total"].item())
            choice_reward_components.append(rewards)
        
        # Find index of correct answer
        correct_index = choices.index(correct_answer) if correct_answer in choices else -1
        
        # Calculate loss: correct answer should have higher reward than others
        if correct_index >= 0:
            correct_reward = choice_rewards[correct_index]
            
            # Calculate pairwise losses with a dynamic margin based on moral clarity
            # The moral clarity is determined by how strongly the explanation indicates
            # the correct choice is better than alternatives
            explanation_strength = 1.0  # Default strength
            if explanation:
                # Estimate explanation strength based on length and key moral terms
                moral_terms = ["ethical", "moral", "right", "good", "better", "should",
                              "appropriate", "responsible", "duty", "obligation"]
                explanation_strength = 1.0 + min(2.0, 0.1 * len(explanation) / 50)
                for term in moral_terms:
                    if term in explanation.lower():
                        explanation_strength += 0.2
            
            # Apply dynamic margin based on explanation strength
            # Stronger explanations lead to larger required margins between correct and incorrect choices
            margin = max(1.0, explanation_strength)
            
            # Calculate pairwise losses
            losses = []
            for i, reward in enumerate(choice_rewards):
                if i != correct_index:
                    # Margin loss: correct_reward should be higher than other_reward by at least margin
                    loss = max(0, margin - (correct_reward - reward))
                    losses.append(loss)
            
            if losses:
                # Weight losses by how wrong each choice is (if mentioned in explanation)
                weighted_losses = []
                for i, loss_val in enumerate(losses):
                    incorrect_idx = i if i < correct_index else i + 1
                    choice_text = choices[incorrect_idx].lower()
                    
                    # Check if this choice is specifically mentioned in explanation as worse
                    weight = 1.0
                    if any(wrong_marker in explanation.lower() for wrong_marker in
                           [choice_text[:20], f"option {incorrect_idx+1}", "first choice" if incorrect_idx == 0 else ""]):
                        weight = 2.0  # Higher weight for explicitly rejected choices
                    
                    weighted_losses.append(loss_val * weight)
                
                # Average the weighted losses
                loss = sum(weighted_losses) / len(weighted_losses)
                
                # Backward pass with stable numerics
                self.optimizer.zero_grad()
                loss_tensor = torch.tensor(loss, device=self.device, requires_grad=True)
                loss_tensor.backward()
                
                # Apply gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.state_encoder.parameters()) +
                    list(self.action_encoder.parameters()),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # Calculate detailed metrics about the reward components
                correct_components = choice_reward_components[correct_index]
                incorrect_components = [comp for i, comp in enumerate(choice_reward_components) if i != correct_index]
                
                # Average the component values across incorrect choices
                avg_incorrect_components = {}
                for key in correct_components.keys():
                    if key != "total":
                        avg_incorrect_components[key] = sum(comp[key].item() for comp in incorrect_components) / len(incorrect_components)
                
                # Return comprehensive metrics including all component-wise gaps for full analysis
                return {
                    "loss": loss,
                    "correct_reward": correct_reward,
                    "avg_incorrect_reward": sum([r for i, r in enumerate(choice_rewards) if i != correct_index]) / (len(choices) - 1),
                    "reward_gap": correct_reward - max([r for i, r in enumerate(choice_rewards) if i != correct_index]),
                    # Complete component-wise gap metrics for the full moral algorithm
                    "mirror_empathy_gap": correct_components["mirror_empathy"].item() - avg_incorrect_components["mirror_empathy"],
                    "env_penalty_gap": correct_components["env_penalty"].item() - avg_incorrect_components["env_penalty"],
                    "emotion_penalty_gap": correct_components["neg_emotion_penalty"].item() - avg_incorrect_components["neg_emotion_penalty"],
                    "dopamine_empathy_gap": correct_components["dopamine_empathy"].item() - avg_incorrect_components["dopamine_empathy"],
                    "self_task_gap": correct_components["self_task"].item() - avg_incorrect_components["self_task"],
                    "perspective_taking_gap": correct_components["perspective_taking"].item() - avg_incorrect_components["perspective_taking"],
                    "episodic_memory_gap": correct_components["episodic_memory"].item() - avg_incorrect_components["episodic_memory"],
                    "altruism_longterm_gap": correct_components["altruism_longterm"].item() - avg_incorrect_components["altruism_longterm"]
                }
        
        # If no correct answer found or no losses
        return {
            "loss": 0.0,
            "correct_reward": 0.0,
            "avg_incorrect_reward": 0.0,
            "reward_gap": 0.0,
            "mirror_empathy_gap": 0.0,
            "env_penalty_gap": 0.0,
            "emotion_penalty_gap": 0.0,
        }
    
    def train(
        self,
        data_path: str = "moral_empathy_dataset.json",
        num_epochs: int = 1,
        batch_size: int = 1,
        save_dir: str = "model_save",
        max_scenarios: int = 800,
        streaming_batch_size: int = 50,
    ) -> List[Dict[str, float]]:
        """
        Train the model on moral choice scenarios loaded in a memory-efficient streaming manner.
        
        Args:
            data_path: Path to the JSON file containing moral choice scenarios
            num_epochs: Number of epochs to train
            batch_size: Training batch size for gradient updates
            save_dir: Directory to save checkpoints
            max_scenarios: Maximum number of scenarios to use for training (default: 800)
            streaming_batch_size: Number of scenarios to load at once from the file
        
        Returns:
            List of training metrics for each epoch
        """
        os.makedirs(save_dir, exist_ok=True)
        epoch_metrics = []
        
        # Function to stream scenarios from the file without loading everything at once
        def stream_scenarios(file_path, start_idx, count):
            scenarios = []
            try:
                with open(file_path, 'r') as f:
                    # Load the JSON structure
                    data = json.load(f)
                    
                    # Extract scenarios based on the data structure
                    if isinstance(data, dict) and "scenarios" in data:
                        all_scenarios = data["scenarios"]
                    elif isinstance(data, list):
                        all_scenarios = data
                    else:
                        print(f"Warning: Unexpected data format in {file_path}")
                        return []
                    
                    # Calculate actual end index (account for file size)
                    end_idx = min(start_idx + count, len(all_scenarios))
                    
                    # Extract the requested slice
                    scenarios = all_scenarios[start_idx:end_idx]
                    
                    print(f"Streamed scenarios {start_idx} to {end_idx-1}")
                    return scenarios, end_idx < len(all_scenarios)
            except Exception as e:
                print(f"Error streaming scenarios from {file_path}: {e}")
                return [], False
        
        # Get total count of scenarios
        total_scenarios_count = 0
        try:
            # Just count the scenarios without storing them
            with open(data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "scenarios" in data:
                    total_scenarios_count = len(data["scenarios"])
                elif isinstance(data, list):
                    total_scenarios_count = len(data)
                else:
                    print(f"Warning: Unexpected data format in {data_path}")
        except Exception as e:
            print(f"Error reading scenario count from {data_path}: {e}")
            return []
        
        # Limit to max_scenarios
        total_scenarios_count = min(total_scenarios_count, max_scenarios)
        print(f"Will train on {total_scenarios_count} scenarios from {data_path}")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()
            
            # Create an array of indices and shuffle them
            all_indices = np.arange(total_scenarios_count)
            np.random.shuffle(all_indices)
            
            # Initialize epoch metrics
            epoch_metric = {
                "epoch": epoch + 1,
                "avg_loss": 0.0,
                "avg_correct_reward": 0.0,
                "avg_incorrect_reward": 0.0,
                "avg_reward_gap": 0.0,
                "num_scenarios": total_scenarios_count
            }
            
            # Track component-wise metrics
            component_metrics = {
                "mirror_empathy_gap": 0.0,
                "env_penalty_gap": 0.0,
                "emotion_penalty_gap": 0.0,
            }
            
            scenarios_processed = 0
            streaming_start = 0
            has_more = True
            
            # Process scenarios in streaming batches to avoid loading all at once
            while scenarios_processed < total_scenarios_count:
                # Load the next batch of scenarios
                scenarios_batch, has_more = stream_scenarios(
                    data_path,
                    streaming_start,
                    streaming_batch_size
                )
                streaming_start += len(scenarios_batch)
                
                if not scenarios_batch:
                    break
                
                # Get shuffled indices for this streaming batch
                batch_indices = all_indices[scenarios_processed:scenarios_processed+len(scenarios_batch)]
                
                # Process training batches within this streaming batch
                for i in range(0, len(batch_indices), batch_size):
                    current_batch_indices = batch_indices[i:i+batch_size]
                    batch_losses = []
                    batch_correct_rewards = []
                    batch_incorrect_rewards = []
                    batch_reward_gaps = []
                    batch_mirror_gaps = []
                    batch_env_gaps = []
                    batch_emotion_gaps = []
                    
                    for idx in current_batch_indices:
                        # Get the actual scenario by its original index
                        actual_idx = idx - scenarios_processed
                        if actual_idx < 0 or actual_idx >= len(scenarios_batch):
                            continue
                            
                        scenario = scenarios_batch[actual_idx]
                        metrics = self.train_on_scenario(scenario)
                        
                        batch_losses.append(metrics["loss"])
                        batch_correct_rewards.append(metrics["correct_reward"])
                        batch_incorrect_rewards.append(metrics["avg_incorrect_reward"])
                        batch_reward_gaps.append(metrics["reward_gap"])
                        
                        # Track component-wise metrics
                        batch_mirror_gaps.append(metrics["mirror_empathy_gap"])
                        batch_env_gaps.append(metrics["env_penalty_gap"])
                        batch_emotion_gaps.append(metrics["emotion_penalty_gap"])
                    
                    # Update epoch metrics with batch results
                    if batch_losses:
                        epoch_metric["avg_loss"] += sum(batch_losses)
                        epoch_metric["avg_correct_reward"] += sum(batch_correct_rewards)
                        epoch_metric["avg_incorrect_reward"] += sum(batch_incorrect_rewards)
                        epoch_metric["avg_reward_gap"] += sum(batch_reward_gaps)
                        
                        # Update component metrics
                        component_metrics["mirror_empathy_gap"] += sum(batch_mirror_gaps)
                        component_metrics["env_penalty_gap"] += sum(batch_env_gaps)
                        component_metrics["emotion_penalty_gap"] += sum(batch_emotion_gaps)
                    
                    # Print progress
                    current_processed = min(scenarios_processed + i + len(current_batch_indices), total_scenarios_count)
                    progress = current_processed / total_scenarios_count * 100
                    print(f"Progress: {progress:.1f}% ({current_processed}/{total_scenarios_count})")
                
                scenarios_processed += len(scenarios_batch)
                
                # Break if we've processed enough scenarios
                if scenarios_processed >= total_scenarios_count:
                    break
                
                # Break if there are no more scenarios to stream
                if not has_more:
                    break
            
            # Adjust for actual number processed
            actually_processed = min(scenarios_processed, total_scenarios_count)
            if actually_processed <= 0:
                print("No scenarios were processed. Check the data file.")
                continue
                
            # Finalize epoch metrics
            epoch_metric["avg_loss"] /= actually_processed
            epoch_metric["avg_correct_reward"] /= actually_processed
            epoch_metric["avg_incorrect_reward"] /= actually_processed
            epoch_metric["avg_reward_gap"] /= actually_processed
            epoch_metric["epoch_time"] = time.time() - epoch_start_time
            
            # Add component metrics to epoch metrics
            epoch_metric["mirror_empathy_gap"] = component_metrics["mirror_empathy_gap"] / actually_processed
            epoch_metric["env_penalty_gap"] = component_metrics["env_penalty_gap"] / actually_processed
            epoch_metric["emotion_penalty_gap"] = component_metrics["emotion_penalty_gap"] / actually_processed
            
            # Save epoch metrics
            epoch_metrics.append(epoch_metric)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} completed in {epoch_metric['epoch_time']:.2f}s")
            print(f"Average loss: {epoch_metric['avg_loss']:.4f}")
            print(f"Average correct reward: {epoch_metric['avg_correct_reward']:.4f}")
            print(f"Average incorrect reward: {epoch_metric['avg_incorrect_reward']:.4f}")
            print(f"Average reward gap: {epoch_metric['avg_reward_gap']:.4f}")
            print(f"Mirror empathy gap: {epoch_metric['mirror_empathy_gap']:.4f}")
            print(f"Environmental penalty gap: {epoch_metric['env_penalty_gap']:.4f}")
            print(f"Emotional penalty gap: {epoch_metric['emotion_penalty_gap']:.4f}")
            
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


def get_stable_loss_function(reduction='mean'):
    """
    Returns a StableCrossEntropyLoss function from GrokOptimizers.
    This loss function is more numerically stable and helps prevent Softmax Collapse.
    
    Args:
        reduction (str): Reduction method, 'mean', 'sum', or 'none'. Default: 'mean'
        
    Returns:
        StableCrossEntropyLoss: A numerically stable loss function
    """
    return StableCrossEntropyLoss(reduction=reduction)

def train_moral_empathy(
    model,
    data_path: str = "moral_empathy_dataset.json",
    embedding_dim: int = 768,
    mirror_weight: float = 0.7,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "model_save",
    entropy_threshold: float = 0.8,
    use_tova: bool = True,
    num_heads: int = 4,
    cache_max_size: int = 512,
    use_stable_loss: bool = True,
    max_scenarios: int = 800,
    streaming_batch_size: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, float]]:
    """
    Train the model on moral empathy using the provided dataset with COCONUT model support.
    Loads data in a memory-efficient streaming fashion and processes up to 800 scenarios.
    
    Args:
        model: Model to be trained (e.g., CoconutBinaryLatentModel)
        data_path: Path to the JSON file containing moral choice scenarios
        embedding_dim: Dimension of state and action embeddings
        mirror_weight: Weight for the mirror neuron empathy component (w_{mirror})
        num_epochs: Number of epochs to train
        batch_size: Batch size for gradient updates
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save checkpoints
        entropy_threshold: Threshold for determining patch boundaries based on entropy
        use_tova: Whether to use TOVA compression for KV caches
        num_heads: Number of attention heads in cross-attention
        cache_max_size: Maximum size of KV cache for TOVA compression
        use_stable_loss: Whether to use numerically stable loss function
        max_scenarios: Maximum number of scenarios to use for training (default: 800)
        streaming_batch_size: Number of scenarios to load at once
        device: Device to run training on
    
    Returns:
        List of training metrics for each epoch
    """
    print(f"Initializing moral empathy training with {max_scenarios} scenarios")
    print(f"Using data file: {data_path}")
    
    # Create moral reward calculator with COCONUT model support
    moral_reward_calculator = FullMoralRewardCalculator(
        embedding_dim=embedding_dim,
        mirror_weight=mirror_weight,
        entropy_threshold=entropy_threshold,
        use_tova=use_tova,
        num_heads=num_heads,
        cache_max_size=cache_max_size,
        device=device
    )
    
    # Create trainer
    trainer = MoralEmpathyTrainer(
        model=model,
        moral_reward_calculator=moral_reward_calculator,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        use_stable_loss=use_stable_loss,
        device=device
    )
    
    # Train model directly from file path with streaming
    metrics = trainer.train(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        max_scenarios=max_scenarios,
        streaming_batch_size=streaming_batch_size
    )
    
    print(f"Completed training on {max_scenarios} scenarios over {num_epochs} epochs")
    
    return metrics