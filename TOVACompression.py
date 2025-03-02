import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

class TOVACompression:
    """
    Token Omission Via Attention (TOVA) with weighted head strategy.
    
    This class implements TOVA compression as described in "Transformers are Multi-State RNNs"
    with enhancements for weighted attention heads and entropy-based token importance scoring.
    
    Key features:
    - Trainable head weights that adapt to the most informative attention heads
    - Entropy-enhanced compression that considers information content of tokens
    - Support for different weighting strategies (mean, max, weighted)
    - Weight visualization and statistic tracking
    """
    
    def __init__(
        self,
        cache_max_size: int = 512,
        layer_based: bool = True,
        head_weight_strategy: str = "mean",
        num_heads: int = 4,
        learning_rate: float = 0.01,
        weight_momentum: float = 0.9,
        entropy_weight: float = 0.3
    ):
        """
        Initialize the TOVA compression module with weighted head strategy.
        
        Args:
            cache_max_size: Maximum number of tokens to keep in the cache
            layer_based: If True, operate on layer-averaged attention weights
            head_weight_strategy: Strategy for combining head weights ('mean', 'max', or 'weighted')
            num_heads: Number of attention heads
            learning_rate: Learning rate for head weight updates
            weight_momentum: Momentum factor for weight updates (0-1)
            entropy_weight: Weight factor for entropy scores (0-1)
        """
        self.cache_max_size = cache_max_size
        self.layer_based = layer_based
        self.head_weight_strategy = head_weight_strategy
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.weight_momentum = weight_momentum
        self.entropy_weight = entropy_weight
        
        # For trainable head weights
        if head_weight_strategy == "weighted":
            # Initialize with equal weights that will be trained
            self.head_weights = nn.Parameter(torch.ones(num_heads))
            # Weight update velocity for momentum-based updates
            self.weight_velocity = torch.zeros(num_heads)
            # Track weight history for visualization
            self.weight_history = []
            self.record_weights()
            # Track token selection statistics for learning
            self.token_selection_history = deque(maxlen=100)
            self.head_contribution_history = deque(maxlen=100)
        
        # Statistics tracking
        self.compression_stats = {
            "total_compressions": 0,
            "tokens_processed": 0,
            "tokens_kept": 0,
            "average_compression_ratio": 0,
            "compression_time_ms": 0,
            "entropy_enhanced_count": 0
        }
    
    def __call__(
        self, 
        attn_weights: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TOVA compression to the KV cache.
        
        Args:
            attn_weights: Attention weights with shape [attn_heads, num_q, num_kv]
            k_cache: Key cache with shape [attn_heads, num_kv, hidden_dim]
            v_cache: Value cache with shape [attn_heads, num_kv, hidden_dim]
            
        Returns:
            Tuple of compressed key and value caches
        """
        start_time = time.time()
        
        # Check if compression is needed
        attn_heads, num_kv = k_cache.shape[0], k_cache.shape[1]
        if num_kv <= self.cache_max_size:
            return k_cache, v_cache
        
        # Get the token scores based on attention weights
        if self.layer_based:
            # Use the attention weights for the last query (most recent token)
            token_scores = self._compute_token_scores(attn_weights[:, -1, :])
        else:
            # Head-wise processing to allow for different tokens per head
            # This is more advanced but requires more complex implementation
            raise NotImplementedError("Head-wise TOVA is not implemented yet")
        
        # Get indices of tokens to keep (highest scores)
        keep_indices = self._get_keep_indices(token_scores, num_kv)
        
        # Update statistics
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["tokens_processed"] += num_kv
        self.compression_stats["tokens_kept"] += self.cache_max_size
        self.compression_stats["average_compression_ratio"] = self.compression_stats["tokens_kept"] / self.compression_stats["tokens_processed"]
        
        # Apply compression
        compressed_k = self._compress_tensor(k_cache, keep_indices)
        compressed_v = self._compress_tensor(v_cache, keep_indices)
        
        # If using weighted head strategy, update weights based on token selection
        if self.head_weight_strategy == "weighted":
            self._update_head_weights(attn_weights[:, -1, :], keep_indices)
            self.record_weights()
        
        # Record compression time
        self.compression_stats["compression_time_ms"] = (time.time() - start_time) * 1000
        
        return compressed_k, compressed_v
    
    def compress_with_entropy(
        self,
        attn_weights: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        entropy_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply entropy-enhanced TOVA compression to the KV cache.
        
        Args:
            attn_weights: Attention weights with shape [attn_heads, num_q, num_kv]
            k_cache: Key cache with shape [attn_heads, num_kv, hidden_dim]
            v_cache: Value cache with shape [attn_heads, num_kv, hidden_dim]
            entropy_values: Entropy scores for tokens with shape [num_kv]
            
        Returns:
            Tuple of compressed key and value caches
        """
        start_time = time.time()
        
        # Check if compression is needed
        attn_heads, num_kv = k_cache.shape[0], k_cache.shape[1]
        if num_kv <= self.cache_max_size:
            return k_cache, v_cache
        
        # Get the token scores based on attention weights
        attn_scores = self._compute_token_scores(attn_weights[:, -1, :])
        
        # Normalize entropy values to range [0, 1]
        if entropy_values.dim() == 1:
            # Make sure entropy_values has the correct size
            if entropy_values.size(0) != num_kv:
                # Handle potential mismatch - entropy might be for a subset of tokens
                if entropy_values.size(0) < num_kv:
                    # Pad with mean value
                    mean_entropy = entropy_values.mean()
                    padded_entropy = torch.full((num_kv,), mean_entropy, 
                                            device=entropy_values.device)
                    padded_entropy[:entropy_values.size(0)] = entropy_values
                    entropy_values = padded_entropy
                else:
                    # Truncate
                    entropy_values = entropy_values[:num_kv]
                    
            # Normalize
            entropy_values = (entropy_values - entropy_values.min()) / (entropy_values.max() - entropy_values.min() + 1e-6)
        
            # Combine attention scores with entropy values
            # Higher entropy means more information, so we want to keep high entropy tokens
            combined_scores = (1 - self.entropy_weight) * attn_scores + self.entropy_weight * entropy_values
        else:
            # If entropy_values is not properly formatted, fall back to attention scores
            combined_scores = attn_scores
        
        # Get indices of tokens to keep (highest scores)
        keep_indices = self._get_keep_indices(combined_scores, num_kv)
        
        # Update statistics
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["tokens_processed"] += num_kv
        self.compression_stats["tokens_kept"] += self.cache_max_size
        self.compression_stats["average_compression_ratio"] = self.compression_stats["tokens_kept"] / self.compression_stats["tokens_processed"]
        self.compression_stats["entropy_enhanced_count"] += 1
        
        # Apply compression
        compressed_k = self._compress_tensor(k_cache, keep_indices)
        compressed_v = self._compress_tensor(v_cache, keep_indices)
        
        # If using weighted head strategy, update weights based on token selection
        if self.head_weight_strategy == "weighted":
            self._update_head_weights(attn_weights[:, -1, :], keep_indices, entropy_values)
            self.record_weights()
        
        # Record compression time
        self.compression_stats["compression_time_ms"] = (time.time() - start_time) * 1000
        
        return compressed_k, compressed_v
    
    def _compute_token_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute token importance scores based on attention weights.
        
        Args:
            attn_weights: Attention weights with shape [attn_heads, num_kv]
            
        Returns:
            Token scores with shape [num_kv]
        """
        if self.head_weight_strategy == "mean":
            # Simple average across heads
            return attn_weights.mean(dim=0)
        
        elif self.head_weight_strategy == "max":
            # Take maximum attention across heads
            return attn_weights.max(dim=0)[0]
        
        elif self.head_weight_strategy == "weighted":
            # Use learnable weights for each head
            # Normalize head weights to sum to 1
            normalized_weights = F.softmax(self.head_weights, dim=0)
            # Apply weights to attention scores
            # Shape: [num_heads, num_kv] * [num_heads, 1] -> [num_heads, num_kv]
            weighted_attn = attn_weights * normalized_weights.unsqueeze(1)
            # Sum over heads to get final scores
            return weighted_attn.sum(dim=0)
        
        else:
            raise ValueError(f"Unknown head weight strategy: {self.head_weight_strategy}")
    
    def _get_keep_indices(self, token_scores: torch.Tensor, num_kv: int) -> torch.Tensor:
        """
        Get indices of tokens to keep based on their importance scores.
        
        Args:
            token_scores: Token importance scores with shape [num_kv]
            num_kv: Number of tokens in the cache
            
        Returns:
            Indices of tokens to keep with shape [cache_max_size]
        """
        # Always keep token 0 (first token)
        token_scores_clone = token_scores.clone()
        
        # Get top-k indices (minus 1 for the reserved first token)
        k = min(self.cache_max_size - 1, num_kv - 1)
        if k <= 0:
            # Edge case: no need to select additional tokens
            return torch.arange(num_kv, device=token_scores.device)
            
        # Set first token's score to max to ensure it's kept
        token_scores_clone[0] = float('inf')
        
        # Get top-k indices
        _, top_indices = torch.topk(token_scores_clone, k=self.cache_max_size)
        
        # Sort indices to maintain token order
        sorted_indices = top_indices.sort()[0]
        
        return sorted_indices
    
    def _compress_tensor(self, tensor: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        """
        Compress a tensor by keeping only the specified indices.
        
        Args:
            tensor: Tensor to compress (k_cache or v_cache)
            keep_indices: Indices of items to keep
            
        Returns:
            Compressed tensor
        """
        if tensor.dim() == 3:  # [num_heads, num_kv, hidden_dim]
            return tensor[:, keep_indices]
        elif tensor.dim() == 2:  # [num_kv, hidden_dim]
            return tensor[keep_indices]
        else:
            raise ValueError(f"Unsupported tensor shape for compression: {tensor.shape}")
    
    def _update_head_weights(
        self, 
        attn_weights: torch.Tensor, 
        keep_indices: torch.Tensor,
        entropy_values: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update the head weights based on their contribution to token selection.
        
        Args:
            attn_weights: Attention weights with shape [attn_heads, num_kv]
            keep_indices: Indices of tokens kept after compression
            entropy_values: Optional entropy values for enhanced learning
        """
        # Skip if not in training mode or no gradients
        if not hasattr(self, 'head_weights') or not self.head_weights.requires_grad:
            return
            
        # Calculate each head's contribution to kept tokens
        num_kv = attn_weights.shape[1]
        
        # Create a mask for kept tokens
        kept_mask = torch.zeros(num_kv, device=attn_weights.device)
        kept_mask[keep_indices] = 1.0
        
        # Calculate contribution scores: how much each head paid attention to kept tokens
        # Higher score = better alignment with token selection
        head_contributions = torch.zeros(self.num_heads, device=attn_weights.device)
        
        for h in range(self.num_heads):
            # Calculate correlation between head's attention pattern and token selection
            head_attn = attn_weights[h]
            
            # Normalize for fair comparison
            if head_attn.max() > head_attn.min():
                head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min())
                
            # Calculate a score based on how well head's attention matches kept tokens
            # Use weighted contribution if entropy values are provided
            if entropy_values is not None and entropy_values.dim() == 1 and entropy_values.size(0) == num_kv:
                # Weight the contribution by entropy (high entropy tokens matter more)
                weighted_match = head_attn * kept_mask * entropy_values
                head_contributions[h] = weighted_match.sum() / (kept_mask * entropy_values).sum()
            else:
                # Standard contribution calculation
                head_contributions[h] = (head_attn * kept_mask).sum() / kept_mask.sum()
        
        # Record for history
        self.token_selection_history.append(kept_mask.detach().cpu())
        self.head_contribution_history.append(head_contributions.detach().cpu())
        
        # Update weights using gradient-like update with momentum
        # Heads that contributed more get higher weights
        with torch.no_grad():
            # Calculate gradient: contribution score relative to current weight
            grad = head_contributions - F.softmax(self.head_weights, dim=0)
            
            # Update velocity with momentum
            self.weight_velocity = self.weight_momentum * self.weight_velocity + (1 - self.weight_momentum) * grad
            
            # Apply update
            self.head_weights.add_(self.learning_rate * self.weight_velocity)
    
    def record_weights(self) -> None:
        """Record current head weights for tracking evolution."""
        if hasattr(self, 'head_weights'):
            # Get normalized weights
            with torch.no_grad():
                weights = F.softmax(self.head_weights, dim=0).detach().cpu().numpy()
            self.weight_history.append(weights)
    
    def plot_weight_evolution(self, save_path: Optional[str] = None) -> bool:
        """
        Plot the evolution of head weights over time.
        
        Args:
            save_path: Path to save the figure (if None, the figure is displayed)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(self, 'weight_history') or len(self.weight_history) < 2:
            return False
            
        try:
            # Convert history to numpy array for plotting
            weights = np.array(self.weight_history)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot each head's weight evolution
            for i in range(weights.shape[1]):
                plt.plot(weights[:, i], label=f"Head {i+1}")
            
            # Add labels and legend
            plt.xlabel("Compression Steps")
            plt.ylabel("Normalized Weight")
            plt.title("Evolution of Attention Head Weights in TOVA Compression")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save or show
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
                
            return True
        except Exception as e:
            print(f"Error plotting weight evolution: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return self.compression_stats
    
    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self.compression_stats = {
            "total_compressions": 0,
            "tokens_processed": 0,
            "tokens_kept": 0,
            "average_compression_ratio": 0,
            "compression_time_ms": 0,
            "entropy_enhanced_count": 0
        }
