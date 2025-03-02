import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import os
import json
import uuid
from collections import OrderedDict
import numpy as np
import faiss
from typing import Optional, List, Dict, Any, Tuple, Union
import shutil
import glob
import pickle
import threading
import math
from concurrent.futures import ThreadPoolExecutor
import warnings

# TOVA Compression (Token Omission Via Attention) for more efficient storage without memory quality loss.
class TOVACompressor:
    """
    Implements TOVA (Token Omission Via Attention) compression algorithm from the paper 
    "Transformers are Multi-State RNNs". Adapted for embeddings rather than tokens.
    
    TOVA selectively maintains important parts of embeddings based on attention scores,
    reducing storage requirements without significant quality loss.
    """
    def __init__(self, 
                embedding_dim: int = 768,
                compression_ratio: float = 0.5,
                attention_heads: int = 8,
                use_layer_wise: bool = True):
        """
        Initialize the TOVA compressor.
        
        Args:
            embedding_dim: Original dimension of embeddings
            compression_ratio: Target compression ratio (0.0-1.0)
            attention_heads: Number of attention heads to use for scoring
            use_layer_wise: Whether to use layer-wise (True) or head-wise (False) attention scoring
        """
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.attention_heads = attention_heads
        self.use_layer_wise = use_layer_wise
        
        # Target dimension after compression
        self.target_dim = max(1, int(embedding_dim * compression_ratio))
        
        # Initialize attention layers for scoring importance
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        
        # Cache for storing attention patterns
        self.attention_cache = {}
        
    def get_attention_scores(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention scores for the embedding dimensions.
        
        Args:
            embedding: Tensor of shape [embedding_dim]
            
        Returns:
            Attention scores for each dimension
        """
        # Reshape to [1, 1, embedding_dim] for attention
        x = embedding.unsqueeze(0).unsqueeze(0)
        
        # Calculate self-attention
        with torch.no_grad():
            attn_output, attn_weights = self.attention(x, x, x)
        
        # attn_weights shape: [1, 1, embedding_dim]
        # Average across attention heads if layer-wise
        if self.use_layer_wise:
            scores = attn_weights.mean(dim=1).squeeze()
        else:
            scores = attn_weights.squeeze()
        
        return scores
    
    def compress(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Compress embedding using TOVA algorithm.
        
        Args:
            embedding: Tensor of shape [embedding_dim]
            
        Returns:
            Tuple of (compressed tensor, indices of kept dimensions)
        """
        # Get attention scores
        scores = self.get_attention_scores(embedding)
        
        # Select top-k dimensions based on attention scores
        if self.use_layer_wise:
            # For layer-wise, we select dimensions with highest overall attention
            _, indices = torch.topk(scores, self.target_dim)
            indices = indices.sort().values  # Sort indices for better cache locality
        else:
            # For head-wise, each head selects dimensions independently
            indices_per_head = []
            dim_per_head = self.target_dim // self.attention_heads
            
            for h in range(self.attention_heads):
                head_scores = scores[h]
                _, head_indices = torch.topk(head_scores, dim_per_head)
                indices_per_head.extend(head_indices.tolist())
            
            # Remove duplicates and take top-k
            indices = torch.tensor(list(set(indices_per_head)), device=embedding.device)
            if len(indices) > self.target_dim:
                _, idx = torch.topk(scores[indices], self.target_dim)
                indices = indices[idx].sort().values
        
        # Extract selected dimensions
        compressed = embedding[indices]
        
        return compressed, indices.tolist()
    
    def decompress(self, compressed: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        Decompress tensor by placing values back in their original positions.
        
        Args:
            compressed: Compressed tensor
            indices: Indices of dimensions that were kept
            
        Returns:
            Decompressed tensor with original dimensionality
        """
        # Create output tensor initialized with zeros
        decompressed = torch.zeros(self.embedding_dim, device=compressed.device)
        
        # Place compressed values back in their original positions
        decompressed[indices] = compressed
        
        return decompressed
    
    def adaptive_compress(self, embedding: torch.Tensor, min_ratio: float = 0.1, 
                          max_ratio: float = 0.9, threshold: float = 0.9) -> Tuple[torch.Tensor, List[int], float]:
        """
        Adaptively compress embedding based on attention distribution.
        
        Args:
            embedding: Tensor to compress
            min_ratio: Minimum compression ratio 
            max_ratio: Maximum compression ratio
            threshold: Attention score threshold for inclusion
            
        Returns:
            Tuple of (compressed tensor, indices, achieved ratio)
        """
        # Get attention scores
        scores = self.get_attention_scores(embedding)
        
        # Normalize scores to 0-1 range
        norm_scores = torch.softmax(scores, dim=0)
        
        # Find dimensions with scores above threshold
        mask = norm_scores > threshold / self.embedding_dim
        
        # Ensure we're within min/max bounds
        min_dims = max(1, int(min_ratio * self.embedding_dim))
        max_dims = min(self.embedding_dim, int(max_ratio * self.embedding_dim))
        
        if mask.sum() < min_dims:
            # Take top min_dims if too few dimensions are selected
            _, indices = torch.topk(norm_scores, min_dims)
        elif mask.sum() > max_dims:
            # Take top max_dims if too many dimensions are selected
            _, indices = torch.topk(norm_scores, max_dims)
        else:
            # Use mask if within bounds
            indices = torch.where(mask)[0]
        
        # Sort indices for better cache locality
        indices = indices.sort().values
        
        # Extract selected dimensions
        compressed = embedding[indices]
        
        # Calculate achieved compression ratio
        achieved_ratio = len(indices) / self.embedding_dim
        
        return compressed, indices.tolist(), achieved_ratio

# Token Omission Via Attention (TOVA) for memory patches
class TOVAPatchCompression:
    """
    Implements TOVA compression for dynamic byte patches used in EpisodicMemory.
    Focuses on reducing memory requirements while maintaining critical information.
    """
    def __init__(self, base_size: int = 768, min_keep_ratio: float = 0.1):
        """
        Initialize TOVA patch compression.
        
        Args:
            base_size: Base size of memory embeddings
            min_keep_ratio: Minimum ratio of data to keep
        """
        self.base_size = base_size
        self.min_keep_ratio = min_keep_ratio
        self.attention_cache = {}
        
    def calculate_importance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Calculate importance scores for memory patches.
        Uses a combination of:
        1. Attention-based scoring (TOVA principle)
        2. Information density (higher variance → more important)
        3. Surprisal (outlier values more important)
        
        Args:
            data: Memory data tensor
            
        Returns:
            Importance scores tensor
        """
        # If 1D tensor, add batch dimension
        if data.dim() == 1:
            data = data.unsqueeze(0)
            
        # Approach 1: Variance-based importance (higher variance = more information)
        var_importance = torch.var(data, dim=1)
        
        # Approach 2: Outlier-based importance (further from mean = more surprising)
        mean = torch.mean(data, dim=1, keepdim=True)
        outlier_importance = torch.abs(data - mean).mean(dim=1)
        
        # Approach 3: Gradient-based importance (approximation of attention)
        # Use autograd to get gradients of the norm w.r.t input
        x = data.detach().clone().requires_grad_(True)
        norm = torch.norm(x, dim=1).sum()
        norm.backward()
        grad_importance = torch.abs(x.grad).mean(dim=1)
        
        # Combine importance metrics
        importance = (var_importance + outlier_importance + grad_importance) / 3
        
        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def compress(self, data: torch.Tensor, target_ratio: float = 0.5) -> Tuple[Dict, torch.Tensor]:
        """
        Compress memory data using TOVA principles.
        
        Args:
            data: Memory data tensor
            target_ratio: Target compression ratio
            
        Returns:
            Tuple of (metadata dict, compressed tensor)
        """
        # Ensure minimum ratio
        ratio = max(self.min_keep_ratio, target_ratio)
        
        # Calculate importance
        importance = self.calculate_importance(data)
        
        # Determine threshold to achieve target ratio
        k = max(1, int(data.shape[0] * ratio))
        if importance.shape[0] <= k:
            # Keep everything if already smaller than target
            kept_indices = torch.arange(importance.shape[0])
            threshold = importance.min().item()
        else:
            # Find threshold for top-k elements
            threshold = torch.topk(importance, k).values[-1].item()
            # Get indices of elements to keep
            kept_indices = torch.where(importance >= threshold)[0]
        
        # Extract data to keep
        compressed_data = data[kept_indices]
        
        # Create metadata for decompression
        metadata = {
            'original_shape': list(data.shape),
            'kept_indices': kept_indices.tolist(),
            'threshold': threshold,
            'compression_ratio': ratio
        }
        
        return metadata, compressed_data
    
    def decompress(self, compressed_data: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Decompress memory data using stored metadata.
        
        Args:
            compressed_data: Compressed data tensor
            metadata: Metadata with decompression information
            
        Returns:
            Decompressed tensor (with zeros in place of omitted values)
        """
        # Create output tensor of original shape filled with zeros
        original_shape = metadata['original_shape']
        decompressed = torch.zeros(original_shape, dtype=compressed_data.dtype,
                                  device=compressed_data.device)
        
        # Place compressed values back in their original positions
        kept_indices = metadata['kept_indices']
        decompressed[kept_indices] = compressed_data
        
        return decompressed
    
    def adaptive_compress(self, data: torch.Tensor, importance_threshold: float = 0.5) -> Tuple[Dict, torch.Tensor]:
        """
        Adaptively compress memory data based on importance threshold.
        
        Args:
            data: Memory data tensor
            importance_threshold: Threshold for keeping data (0.0-1.0)
            
        Returns:
            Tuple of (metadata dict, compressed tensor)
        """
        # Calculate importance
        importance = self.calculate_importance(data)
        
        # Get indices of elements to keep based on threshold
        kept_indices = torch.where(importance >= importance_threshold)[0]
        
        # Ensure we keep at least some minimum amount
        min_keep = max(1, int(data.shape[0] * self.min_keep_ratio))
        if len(kept_indices) < min_keep:
            # If too few elements selected, take top-k by importance
            _, top_indices = torch.topk(importance, min_keep)
            kept_indices = top_indices
        
        # Extract data to keep
        compressed_data = data[kept_indices]
        
        # Achieved compression ratio
        achieved_ratio = len(kept_indices) / data.shape[0]
        
        # Create metadata for decompression
        metadata = {
            'original_shape': list(data.shape),
            'kept_indices': kept_indices.tolist(),
            'threshold': importance_threshold,
            'compression_ratio': achieved_ratio
        }
        
        return metadata, compressed_data

# Based on Titans: Learning to Memorize at Test Time
class NeuralMemoryModule(nn.Module):
    """
    Neural long-term memory module as described in the Titans paper.
    This module learns to memorize at test time using a deep neural network.
    Additionally implements TOVA compression for efficient memory storage.
    """
    def __init__(self, 
                 embedding_dim: int = 768, 
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 surprise_threshold: float = 0.7,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 forget_rate: float = 0.1,
                 device=None,
                 use_tova: bool = True,
                 save_path: str = None,
                 auto_save: bool = True):
        """
        Initialize the neural memory module.
        
        Args:
            embedding_dim: Dimension of the input/output embeddings
            hidden_dim: Hidden dimension for the MLP layers
            num_layers: Number of layers in the MLP
            surprise_threshold: Threshold for determining if an input is surprising
            learning_rate: Learning rate for memory updates
            momentum: Momentum factor for surprise tracking
            forget_rate: Base rate for adaptive forgetting
            device: Device to place the model on
            use_tova: Whether to use TOVA compression
            save_path: Path to save persistent TOVA patterns
        """
        super(NeuralMemoryModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.surprise_threshold = surprise_threshold
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.forget_rate = forget_rate
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tova = use_tova
        self.save_path = save_path or "model_save/neural_memory_tova"
        
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        # Previous surprise state (for momentum)
        self.previous_surprise = None
        
        # For adaptive forgetting
        self.forget_gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # TOVA compression components
        if use_tova:
            # Initialize TOVA attention for determining important dimensions
            self.tova_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                batch_first=True
            )
            
            # Track persistent TOVA importance patterns
            # This represents the accumulated understanding of which dimensions are important
            self.register_buffer('tova_importance', torch.ones(embedding_dim))
            
            # TOVA compressed representations dictionary - maps memory IDs to compressed data
            self.tova_compressed_memories = {}
            
            # TOVA compression ratios by importance level
            self.tova_compression_ratios = {
                'light': 0.9,   # 90% of dimensions
                'medium': 0.5,  # 50% of dimensions
                'heavy': 0.2,   # 20% of dimensions
                'extreme': 0.1  # 10% of dimensions
            }
            
            # Flag to track if a save is scheduled
            self._tova_save_scheduled = False
            
            # Whether to automatically save patterns
            self.auto_save = auto_save
            
            # Load TOVA patterns if they exist
            self._load_tova_patterns()
        
        # Move to device
        self.to(self.device)
    
    def _load_tova_patterns(self):
        """Load persistent TOVA patterns from disk if they exist."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        tova_path = f"{self.save_path}_importance.pt"
        cached_memories_path = f"{self.save_path}_memories.pkl"
        
        if os.path.exists(tova_path):
            try:
                self.tova_importance = torch.load(tova_path, map_location=self.device)
                print(f"Loaded persistent TOVA importance patterns from {tova_path}")
            except Exception as e:
                print(f"Error loading TOVA importance patterns: {e}")
                # Initialize with ones if loading fails
                self.tova_importance = torch.ones(self.embedding_dim, device=self.device)
        else:
            # Initialize with ones if file doesn't exist
            self.tova_importance = torch.ones(self.embedding_dim, device=self.device)
            print(f"Initialized new TOVA importance patterns (no existing file found)")
        
        if os.path.exists(cached_memories_path):
            try:
                with open(cached_memories_path, 'rb') as f:
                    self.tova_compressed_memories = pickle.load(f)
                print(f"Loaded {len(self.tova_compressed_memories)} cached TOVA compressed memories")
            except Exception as e:
                print(f"Error loading cached TOVA memories: {e}")
                self.tova_compressed_memories = {}
        else:
            self.tova_compressed_memories = {}
            print("No cached TOVA memories found, initialized empty dictionary")
    
    def _save_tova_patterns(self):
        """Save persistent TOVA patterns to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        tova_path = f"{self.save_path}_importance.pt"
        cached_memories_path = f"{self.save_path}_memories.pkl"
        
        try:
            # Save TOVA importance patterns
            torch.save(self.tova_importance.detach().cpu(), tova_path)
            
            # Save cached compressed memories
            # Limit to most recently used 1000 items to prevent excessive storage
            if len(self.tova_compressed_memories) > 1000:
                # Sort by last accessed time and keep most recent 1000
                sorted_items = sorted(
                    self.tova_compressed_memories.items(),
                    key=lambda x: x[1].get('last_accessed', 0),
                    reverse=True
                )[:1000]
                self.tova_compressed_memories = dict(sorted_items)
            
            with open(cached_memories_path, 'wb') as f:
                pickle.dump(self.tova_compressed_memories, f)
                
            print(f"Saved TOVA patterns to {tova_path} and {cached_memories_path}")
            
            # Reset the scheduled flag after saving
            if hasattr(self, '_tova_save_scheduled'):
                self._tova_save_scheduled = False
        except Exception as e:
            print(f"Error saving TOVA patterns: {e}")
            # Reset the scheduled flag even on error to avoid blocking future saves
            if hasattr(self, '_tova_save_scheduled'):
                self._tova_save_scheduled = False
    
    def update_tova_importance(self, embedding: torch.Tensor):
        """
        Update the persistent TOVA importance patterns based on a new embedding.
        This allows the model to learn over time which dimensions are most important.
        
        Args:
            embedding: New embedding to analyze
        """
        if not self.use_tova:
            return
            
        with torch.no_grad():
            # Reshape for attention
            x = embedding.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Calculate self-attention
            attn_output, attn_weights = self.tova_attention(x, x, x)
            
            # Average across attention heads
            current_importance = attn_weights.mean(dim=1).squeeze()
            
            # Update persistent importance with exponential moving average
            alpha = 0.05  # Update rate (low for stability)
            self.tova_importance = (1 - alpha) * self.tova_importance + alpha * current_importance
            
            # Schedule saving the updated patterns
            if self.auto_save and hasattr(self, '_tova_save_scheduled') and not self._tova_save_scheduled:
                self._tova_save_scheduled = True
                # Save asynchronously to avoid blocking
                threading.Thread(target=self._save_tova_patterns).start()
    
    def save_on_shutdown(self):
        """
        Save TOVA patterns and compressed memories on shutdown.
        This ensures all memory is properly persisted when the program is terminated.
        """
        if not self.use_tova:
            return
            
        # Force immediate save regardless of scheduled status
        print("Saving TOVA patterns and compressed memories on shutdown...")
        self._save_tova_patterns()
        print("Memory persistence complete.")
    
    def get_important_dimensions(self, level: str = 'medium') -> List[int]:
        """
        Get the most important dimensions based on persistent TOVA patterns.
        
        Args:
            level: Compression level determining how many dimensions to return
            
        Returns:
            List of indices of the most important dimensions
        """
        if not self.use_tova:
            # If TOVA not enabled, return evenly spaced dimensions
            ratio = self.tova_compression_ratios.get(level, 0.5)
            target_dim = max(1, int(self.embedding_dim * ratio))
            return list(range(0, self.embedding_dim, self.embedding_dim // target_dim))[:target_dim]
        
        # Get target dimension count based on level
        ratio = self.tova_compression_ratios.get(level, 0.5)
        target_dim = max(1, int(self.embedding_dim * ratio))
        
        # Get top dimensions based on importance
        _, indices = torch.topk(self.tova_importance, target_dim)
        return indices.tolist()
    
    def tova_compress(self, memory_id: str, embedding: torch.Tensor, level: str = 'medium') -> Dict:
        """
        Compress an embedding using TOVA and store in the cached memory.
        
        Args:
            memory_id: Unique ID for this memory
            embedding: Embedding tensor to compress
            level: Compression level
            
        Returns:
            Compressed data dictionary
        """
        if not self.use_tova:
            return {'type': 'none', 'tensor': embedding.detach().cpu().numpy()}
        
        # Update importance patterns with this embedding
        self.update_tova_importance(embedding)
        
        # Get important dimensions for this level
        important_dims = self.get_important_dimensions(level)
        
        # Extract values at important dimensions
        compressed_tensor = embedding[important_dims]
        
        # Create compressed data structure
        compressed_data = {
            'type': 'tova',
            'tensor': compressed_tensor.detach().cpu().numpy(),
            'indices': important_dims,
            'level': level,
            'embedding_dim': self.embedding_dim,
            'last_accessed': time.time()
        }
        
        # Store in cache
        self.tova_compressed_memories[memory_id] = compressed_data
        
        # Schedule save if many new memories added
        if len(self.tova_compressed_memories) % 50 == 0:
            self._save_tova_patterns()
        
        return compressed_data
    
    def tova_decompress(self, compressed_data: Dict) -> torch.Tensor:
        """
        Decompress a TOVA-compressed embedding.
        
        Args:
            compressed_data: Compressed data dictionary
            
        Returns:
            Decompressed embedding tensor
        """
        if not isinstance(compressed_data, dict) or compressed_data.get('type') != 'tova':
            # Not a TOVA compression or invalid format
            if isinstance(compressed_data, dict) and 'tensor' in compressed_data:
                return torch.tensor(compressed_data['tensor'], device=self.device)
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Extract compressed tensor and indices
        tensor = compressed_data['tensor']
        if isinstance(tensor, np.ndarray):
            tensor = torch.tensor(tensor, device=self.device)
        indices = compressed_data['indices']
        
        # Create output tensor filled with zeros
        decompressed = torch.zeros(self.embedding_dim, device=self.device)
        
        # Place compressed values back in their original positions
        decompressed[indices] = tensor
        
        return decompressed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get memory corresponding to the input.
        
        Args:
            x: Input tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Memory output tensor of shape [batch_size, embedding_dim]
        """
        # Memory retrieval
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
        return self.layers[-1](h)
    
    def calculate_surprise(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate surprise based on prediction error.
        
        Args:
            x: Input tensor
            target: Target tensor
            
        Returns:
            Surprise level tensor
        """
        # Forward pass to get prediction
        prediction = self.forward(x)
        
        # Calculate loss
        loss = F.mse_loss(prediction, target, reduction='none').mean(dim=-1)
        
        # Get gradients of loss with respect to model parameters
        gradients = []
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Calculate gradients for each example in the batch
        for i in range(x.size(0)):
            loss[i].backward(retain_graph=True)
            
            # Collect gradients
            grad_norm = 0
            for param in self.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            
            gradients.append(grad_norm)
            
            # Zero gradients for next iteration
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Convert to tensor
        momentary_surprise = torch.tensor(gradients, device=x.device)
        
        return momentary_surprise
    
    def update(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Update memory parameters based on the input.
        
        Args:
            x: Input tensor
            target: Target tensor (if None, uses x as target for auto-associative memory)
            
        Returns:
            Surprise level tensor
        """
        if target is None:
            target = x
        
        # Calculate momentary surprise
        momentary_surprise = self.calculate_surprise(x, target)
        
        # Update with momentum to track surprise over time
        if self.previous_surprise is None:
            self.previous_surprise = torch.zeros_like(momentary_surprise)
        
        surprise = self.momentum * self.previous_surprise - (1 - self.momentum) * momentary_surprise
        self.previous_surprise = surprise.detach()
        
        # Calculate adaptive forget rate
        adaptive_forget_rate = self.forget_gate(x).squeeze(-1)
        effective_forget_rate = self.forget_rate * adaptive_forget_rate
        
        # Update memory parameters based on surprise and forget rate
        output = self.forward(x)
        loss = F.mse_loss(output, target)
        
        # Calculate gradients
        gradients = torch.autograd.grad(loss, self.parameters(), create_graph=False)
        
        # Update parameters with adaptive forgetting
        with torch.no_grad():
            for i, (param, grad) in enumerate(zip(self.parameters(), gradients)):
                param.data = (1 - effective_forget_rate.mean()) * param.data - self.learning_rate * grad
        
        return surprise

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the next state based on current memory state.
        Used for surprise calculation in advanced settings.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted next state
        """
        return self.forward(x)

class MemoryItem:
    """
    A class to represent an individual memory item for the episodic memory system.
    """
    def __init__(self, 
                 embedding: torch.Tensor, 
                 timestamp: float = None, 
                 surprise_level: float = 0.0,
                 agent_info_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a memory item.
        
        Args:
            embedding: The embedding tensor representing the memory content
            timestamp: The time when the memory was created (defaults to current time)
            surprise_level: The level of surprise associated with this memory
            agent_info_id: Optional identifier for an agent associated with this memory
            metadata: Additional metadata associated with the memory
        """
        self.id = str(uuid.uuid4())
        self.embedding = embedding
        self.timestamp = timestamp if timestamp else time.time()
        self.surprise_level = surprise_level
        self.agent_info_id = agent_info_id
        self.metadata = metadata or {}
        self.last_accessed = time.time()
        self.access_count = 1  # How many times this memory has been accessed
    
    def mark_accessed(self):
        """Update the last accessed timestamp and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self):
        """
        Convert memory item to a dictionary for serialization.
        Note: embedding is excluded as it's handled separately.
        """
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'surprise_level': self.surprise_level,
            'agent_info_id': self.agent_info_id,
            'metadata': self.metadata,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data, embedding):
        """
        Create a memory item from a dictionary and embedding.
        
        Args:
            data: Dictionary containing memory item data
            embedding: The embedding tensor
            
        Returns:
            A MemoryItem instance
        """
        item = cls(
            embedding=embedding,
            timestamp=data['timestamp'],
            surprise_level=data['surprise_level'],
            agent_info_id=data['agent_info_id'],
            metadata=data['metadata']
        )
        item.id = data['id']
        item.last_accessed = data.get('last_accessed', time.time())
        item.access_count = data.get('access_count', 1)
        return item

class LRUCache:
    """
    A Least Recently Used (LRU) cache implementation.
    """
    def __init__(self, capacity: int):
        """
        Initialize an LRU cache with the specified capacity.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """
        Retrieve an item from the cache by key.
        
        Args:
            key: The cache key to look up
            
        Returns:
            The cached value or None if not found
        """
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        """
        Add an item to the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class PersistentMemory(nn.Module):
    """
    Persistent memory module as described in the Titans paper.
    This represents task-specific input-independent memory with learnable parameters.
    """
    def __init__(self, embedding_dim: int = 768, num_items: int = 32):
        """
        Initialize the persistent memory module.
        
        Args:
            embedding_dim: Dimension of memory embeddings
            num_items: Number of persistent memory items
        """
        super(PersistentMemory, self).__init__()
        
        # Learnable persistent memory parameters
        self.memory = nn.Parameter(torch.randn(num_items, embedding_dim) * 0.02)
        
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get the persistent memory items.
        
        Args:
            batch_size: Batch size for expansion
            
        Returns:
            Persistent memory tensor of shape [batch_size, num_items, embedding_dim]
        """
        return self.memory.unsqueeze(0).expand(batch_size, -1, -1)

class MemoryIntegration(nn.Module):
    """
    Implements the three Titans memory integration architectures:
    - Memory as Context (MAC)
    - Memory as Gate (MAG)
    - Memory as Layer (MAL)
    """
    def __init__(self, 
                 embedding_dim: int = 768, 
                 integration_type: str = "MAC",
                 num_heads: int = 8):
        """
        Initialize the memory integration module.
        
        Args:
            embedding_dim: Dimension of memory embeddings
            integration_type: Type of memory integration ('MAC', 'MAG', or 'MAL')
            num_heads: Number of attention heads for MAC and MAL
        """
        super(MemoryIntegration, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.integration_type = integration_type
        
        if integration_type == "MAC":
            # Memory as Context uses attention to process combined input
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim, 
                num_heads=num_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(embedding_dim)
            
        elif integration_type == "MAG":
            # Memory as Gate uses gating to combine memory and context
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            self.transform = nn.Linear(embedding_dim * 2, embedding_dim)
            
        elif integration_type == "MAL":
            # Memory as Layer passes input through memory first, then attention
            self.pre_norm = nn.LayerNorm(embedding_dim)
            self.post_norm = nn.LayerNorm(embedding_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
        else:
            raise ValueError(f"Unknown integration type: {integration_type}")
    
    def forward(self, x: torch.Tensor, memory_context: torch.Tensor) -> torch.Tensor:
        """
        Integrate memory context with input using the specified architecture.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            memory_context: Memory context tensor of shape [batch_size, context_len, embedding_dim]
            
        Returns:
            Integrated output tensor
        """
        if self.integration_type == "MAC":
            # Memory as Context: concatenate memory context with input and process with attention
            combined = torch.cat([memory_context, x], dim=1)
            attn_output, _ = self.attention(x, combined, combined)
            return self.norm(x + attn_output)
            
        elif self.integration_type == "MAG":
            # Memory as Gate: use gating mechanism to combine input and memory
            # First, get a single memory context vector per example
            if memory_context.size(1) > 1:
                memory_context = memory_context.mean(dim=1, keepdim=True)
            
            # Expand memory context to match input sequence length
            memory_context = memory_context.expand(-1, x.size(1), -1)
            
            # Concatenate and compute gate values
            combined = torch.cat([x, memory_context], dim=-1)
            gate = self.gate(combined)
            
            # Apply gate
            transformed = self.transform(combined)
            return gate * transformed + (1 - gate) * x
            
        elif self.integration_type == "MAL":
            # Memory as Layer: process input with memory as a layer, then apply attention
            normalized = self.pre_norm(x)
            attn_output, _ = self.attention(normalized, memory_context, memory_context)
            return self.post_norm(x + attn_output)

class LifetimeMemoryStats:
    """
    Track statistics for the lifetime memory system for monitoring and optimization.
    """
    def __init__(self):
        self.total_memories_added = 0  # Total number of memories ever added
        self.total_memories_archived = 0  # Total memories moved to archive
        self.total_memories_retrieved = 0  # Total memories retrieved from archive
        self.active_memory_count = 0  # Current active memory count
        self.archive_memory_count = 0  # Current archive memory count
        
        # Performance tracking
        self.avg_retrieval_time = 0  # Average time for memory retrieval
        self.retrieval_count = 0  # Number of retrievals performed
        
        # Time-based statistics
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.archive_stats = {}  # Stats per time period {"YYYY_MM": count}
        
    def update_retrieval_time(self, retrieval_time):
        """Update average retrieval time with a new sample."""
        self.avg_retrieval_time = (self.avg_retrieval_time * self.retrieval_count + retrieval_time) / (self.retrieval_count + 1)
        self.retrieval_count += 1
        self.last_access_time = time.time()
    
    def record_memory_added(self):
        """Record that a memory was added to active memory."""
        self.total_memories_added += 1
        self.active_memory_count += 1
        self.last_access_time = time.time()
    
    def record_memory_archived(self, year_month):
        """Record that a memory was archived."""
        self.total_memories_archived += 1
        self.active_memory_count -= 1
        self.archive_memory_count += 1
        self.archive_stats[year_month] = self.archive_stats.get(year_month, 0) + 1
        self.last_access_time = time.time()
    
    def record_memory_retrieved(self):
        """Record that a memory was retrieved from archive."""
        self.total_memories_retrieved += 1
        self.last_access_time = time.time()
    
    def to_dict(self):
        """Convert stats to a dictionary for serialization."""
        return {
            'total_memories_added': self.total_memories_added,
            'total_memories_archived': self.total_memories_archived,
            'total_memories_retrieved': self.total_memories_retrieved,
            'active_memory_count': self.active_memory_count,
            'archive_memory_count': self.archive_memory_count,
            'avg_retrieval_time': self.avg_retrieval_time,
            'retrieval_count': self.retrieval_count,
            'creation_time': self.creation_time,
            'last_access_time': self.last_access_time,
            'archive_stats': self.archive_stats,
            'age_in_days': (time.time() - self.creation_time) / (24 * 3600)
        }

class MemoryCompressionService:
    """
    Service for compressing and decompressing memory embeddings.
    Uses a combination of dimensionality reduction, attention-based TOVA compression,
    and quantization techniques to reduce storage requirements while maintaining quality.
    """
    def __init__(self, 
                 embedding_dim: int = 768,
                 compression_levels: Dict[str, float] = None,
                 use_tova: bool = True,
                 attention_heads: int = 8,
                 neural_memory: Optional['NeuralMemoryModule'] = None):
        """
        Initialize the memory compression service.
        
        Args:
            embedding_dim: Original dimension of embeddings
            compression_levels: Dictionary mapping level names to compression factors
                               (e.g., {'light': 0.8, 'medium': 0.5, 'heavy': 0.2})
            use_tova: Whether to use TOVA compression (Token Omission Via Attention)
            attention_heads: Number of attention heads for TOVA
            neural_memory: Optional neural memory module to use for persistent TOVA patterns
        """
        self.embedding_dim = embedding_dim
        self.compression_levels = compression_levels or {
            'light': 0.9,  # 90% of original size
            'medium': 0.5,  # 50% of original size
            'heavy': 0.2,   # 20% of original size
            'extreme': 0.1  # 10% of original size
        }
        self.use_tova = use_tova
        self.neural_memory = neural_memory
        
        # Create PCA models for each compression level
        self.pca_models = {}
        self.is_trained = {}
        
        for level, factor in self.compression_levels.items():
            # Calculate target dimension for this level
            target_dim = max(1, int(embedding_dim * factor))
            self.pca_models[level] = None  # Will be initialized when trained
            self.is_trained[level] = False
        
        # Initialize TOVA compressors with different compression ratios
        self.tova_compressors = {}
        if use_tova:
            for level, factor in self.compression_levels.items():
                self.tova_compressors[level] = TOVACompressor(
                    embedding_dim=embedding_dim,
                    compression_ratio=factor,
                    attention_heads=attention_heads,
                    use_layer_wise=True  # Layer-wise generally performs better
                )
            
            # Also initialize the patch-based TOVA compressor for special cases
            self.tova_patch_compressor = TOVAPatchCompression(
                base_size=embedding_dim
            )
            
        # Quantization parameters for different levels
        # For embeddings, we can use different bit precision
        self.quantization_bits = {
            'light': 32,  # Full precision (float32)
            'medium': 16,  # Half precision (float16)
            'heavy': 8,   # 8-bit quantization
            'extreme': 4  # 4-bit quantization
        }
    
    def train(self, embeddings: torch.Tensor, level: str = 'medium'):
        """
        Train compression model on a batch of embeddings.
        
        Args:
            embeddings: Tensor of shape [n, embedding_dim]
            level: Compression level to train
        """
        if level not in self.compression_levels:
            raise ValueError(f"Unknown compression level: {level}")
        
        # Convert to numpy for PCA
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Calculate target dimension
        target_dim = max(1, int(self.embedding_dim * self.compression_levels[level]))
        
        # Initialize and fit PCA
        from sklearn.decomposition import PCA
        self.pca_models[level] = PCA(n_components=target_dim)
        self.pca_models[level].fit(embeddings_np)
        self.is_trained[level] = True
    
    def compress(self, embedding: torch.Tensor, level: str = 'medium', memory_id: str = None) -> bytes:
        """
        Compress a single embedding using TOVA or traditional methods.
        
        Args:
            embedding: Tensor of shape [embedding_dim]
            level: Compression level
            memory_id: Optional memory ID for persistent TOVA caching
            
        Returns:
            Compressed embedding as bytes
        """
        if level not in self.compression_levels:
            raise ValueError(f"Unknown compression level: {level}")
        
        # Use TOVA compression if enabled
        if self.use_tova and level in self.tova_compressors:
            # Option 1: If neural memory is connected, use its TOVA patterns
            if self.neural_memory and self.neural_memory.use_tova:
                # Use neural memory's learned importance patterns
                important_dims = self.neural_memory.get_important_dimensions(level)
                
                # Extract values at important dimensions
                compressed_tensor = embedding[important_dims]
                
                # Update neural memory's importance patterns with this embedding
                self.neural_memory.update_tova_importance(embedding)
                
                # If memory_id provided, store in neural memory's cache
                if memory_id:
                    compressed_data_dict = self.neural_memory.tova_compress(memory_id, embedding, level)
                    
                    # Create serializable compressed data
                    compressed_data = {
                        'type': 'tova',
                        'tensor': compressed_data_dict['tensor'],
                        'indices': compressed_data_dict['indices'],
                        'level': level,
                        'embedding_dim': self.embedding_dim,
                        'compression_ratio': self.compression_levels[level],
                        'memory_id': memory_id
                    }
                else:
                    # Create serializable compressed data
                    compressed_data = {
                        'type': 'tova',
                        'tensor': compressed_tensor.detach().cpu().numpy(),
                        'indices': important_dims,
                        'level': level,
                        'embedding_dim': self.embedding_dim,
                        'compression_ratio': self.compression_levels[level]
                    }
            else:
                # Option 2: Use standalone TOVA compressor
                compressed_tensor, indices = self.tova_compressors[level].compress(embedding)
                
                compressed_data = {
                    'type': 'tova',
                    'tensor': compressed_tensor.detach().cpu().numpy(),
                    'indices': indices,
                    'level': level,
                    'embedding_dim': self.embedding_dim,
                    'compression_ratio': self.compression_levels[level]
                }
            
            # Apply additional quantization for heavy/extreme compression levels
            if level == 'heavy':
                # Convert to 8-bit precision
                compressed_data['tensor'] = self._quantize_array(compressed_data['tensor'], 8)
                compressed_data['quantized'] = True
                compressed_data['bits'] = 8
            elif level == 'extreme':
                # Convert to 4-bit precision
                compressed_data['tensor'] = self._quantize_array(compressed_data['tensor'], 4)
                compressed_data['quantized'] = True
                compressed_data['bits'] = 4
        else:
            # Fall back to traditional compression if TOVA not enabled or initialized
            if not self.is_trained[level]:
                # For initial compression, we'll use a simpler method
                if level == 'light':
                    # Light compression just uses float16
                    compressed_data = {
                        'type': 'float16',
                        'tensor': embedding.detach().cpu().to(torch.float16).numpy(),
                        'level': level
                    }
                elif level == 'medium':
                    # Medium compression uses float16 and keeps half the values
                    tensor = embedding.detach().cpu().to(torch.float16).numpy()
                    target_dim = max(1, int(self.embedding_dim * self.compression_levels[level]))
                    compressed_data = {
                        'type': 'truncated',
                        'tensor': tensor[:target_dim],
                        'original_dim': self.embedding_dim,
                        'level': level
                    }
                elif level in ['heavy', 'extreme']:
                    # Heavy/extreme compression quantizes to 8/4 bit
                    bits = 8 if level == 'heavy' else 4
                    quantized = self._quantize(embedding, bits)
                    compressed_data = {
                        'type': 'quantized',
                        'tensor': quantized,
                        'level': level,
                        'bits': bits
                    }
            else:
                # Use PCA for compression
                embedding_np = embedding.detach().cpu().numpy().reshape(1, -1)
                compressed = self.pca_models[level].transform(embedding_np)[0]
                
                compressed_data = {
                    'type': 'pca',
                    'tensor': compressed,
                    'level': level
                }
                
                # Apply additional quantization for heavy and extreme
                if level == 'heavy':
                    compressed_data['tensor'] = self._quantize_array(compressed_data['tensor'], 8)
                    compressed_data['quantized'] = True
                    compressed_data['bits'] = 8
                elif level == 'extreme':
                    compressed_data['tensor'] = self._quantize_array(compressed_data['tensor'], 4)
                    compressed_data['quantized'] = True
                    compressed_data['bits'] = 4
        
        # Serialize the compressed data with metadata
        return pickle.dumps(compressed_data)
    
    def decompress(self, compressed_bytes: bytes, level: str = 'medium') -> torch.Tensor:
        """
        Decompress a compressed embedding using the same method as was used to compress.
        
        Args:
            compressed_bytes: Compressed embedding bytes
            level: Compression level that was used (as fallback if not specified in data)
            
        Returns:
            Decompressed embedding tensor (may be lossy)
        """
        # Load the compressed data with metadata
        compressed_data = pickle.loads(compressed_bytes)
        
        # Check if the compressed data is in the new structured format
        if isinstance(compressed_data, dict) and 'type' in compressed_data:
            compression_type = compressed_data['type']
            
            # Handle TOVA compressed data
            if compression_type == 'tova' and self.use_tova:
                # Prepare for TOVA decompression
                indices = compressed_data['indices']
                tensor_data = compressed_data['tensor']
                embedding_dim = compressed_data.get('embedding_dim', self.embedding_dim)
                
                # Check if the tensor was quantized
                if compressed_data.get('quantized', False):
                    bits = compressed_data.get('bits', 8)
                    # Dequantize first
                    if isinstance(tensor_data, dict) and 'data' in tensor_data:
                        # This is our quantized format
                        tensor = self._dequantize(tensor_data, bits)
                    else:
                        # Handle other quantization formats
                        warnings.warn("Unknown quantization format in TOVA compressed data")
                        tensor = torch.tensor(tensor_data, dtype=torch.float32)
                else:
                    # Regular tensor data
                    tensor = torch.tensor(tensor_data, dtype=torch.float32)
                
                # Use the appropriate TOVA compressor
                if embedding_dim in [self.embedding_dim, len(indices) + tensor.shape[0]]:
                    # We can use the cached TOVA compressor for this level
                    ratio = compressed_data.get('compression_ratio', self.compression_levels[level])
                    tova_level = level
                    for l, r in self.compression_levels.items():
                        if abs(r - ratio) < abs(self.compression_levels[tova_level] - ratio):
                            tova_level = l
                    
                    if tova_level in self.tova_compressors:
                        # Use the existing compressor to decompress
                        return self.tova_compressors[tova_level].decompress(tensor, indices)
                    else:
                        # Create a temporary compressor
                        temp_compressor = TOVACompressor(embedding_dim=embedding_dim, 
                                                         compression_ratio=ratio)
                        return temp_compressor.decompress(tensor, indices)
                else:
                    # Direct decompression - create tensor and fill values
                    decompressed = torch.zeros(embedding_dim, dtype=tensor.dtype)
                    decompressed[indices] = tensor
                    return decompressed
            
            # Handle float16 compression
            elif compression_type == 'float16':
                return torch.tensor(compressed_data['tensor'], dtype=torch.float32)
            
            # Handle truncated compression
            elif compression_type == 'truncated':
                tensor = compressed_data['tensor']
                original_dim = compressed_data.get('original_dim', self.embedding_dim)
                restored = np.zeros(original_dim, dtype=np.float32)
                restored[:len(tensor)] = tensor
                return torch.tensor(restored, dtype=torch.float32)
            
            # Handle quantized compression
            elif compression_type == 'quantized':
                bits = compressed_data.get('bits', 8)
                return self._dequantize(compressed_data['tensor'], bits)
                
            # Handle PCA compression
            elif compression_type == 'pca':
                tensor = compressed_data['tensor']
                
                # Dequantize if quantized
                if compressed_data.get('quantized', False):
                    bits = compressed_data.get('bits', 8)
                    tensor = self._dequantize(tensor, bits)
                    tensor_np = tensor.numpy()
                else:
                    tensor_np = tensor
                
                # Use PCA to decompress
                if self.is_trained[level]:
                    decompressed = self.pca_models[level].inverse_transform(tensor_np.reshape(1, -1))[0]
                    return torch.tensor(decompressed, dtype=torch.float32)
                else:
                    # If PCA model not trained, just pad with zeros
                    warnings.warn(f"PCA model for level {level} not trained, padding with zeros instead")
                    restored = np.zeros(self.embedding_dim, dtype=np.float32)
                    restored[:len(tensor_np)] = tensor_np
                    return torch.tensor(restored, dtype=torch.float32)
            
            # Unknown compression type
            else:
                warnings.warn(f"Unknown compression type: {compression_type}, using fallback")
        
        # Legacy format or simple data structure
        # This handles backward compatibility with old compression format
        if not self.is_trained[level]:
            # For simple compression methods
            if level == 'light':
                # Just convert back to full precision
                return torch.tensor(compressed_data, dtype=torch.float32)
            elif level == 'medium':
                # Pad with zeros to restore original dimension
                restored = np.zeros(self.embedding_dim, dtype=np.float32)
                # Handle both array-like and dictionary formats
                if isinstance(compressed_data, dict) and 'data' in compressed_data:
                    data_to_restore = compressed_data['data']
                else:
                    data_to_restore = compressed_data
                restored[:len(data_to_restore)] = data_to_restore
                return torch.tensor(restored, dtype=torch.float32)
            elif level in ['heavy', 'extreme']:
                # Dequantize
                bits = 8 if level == 'heavy' else 4
                return self._dequantize(compressed_data, bits)
        else:
            # Use PCA for decompression
            try:
                # Handle both array-like and dictionary formats
                if isinstance(compressed_data, dict) and 'data' in compressed_data:
                    data_to_decompress = compressed_data['data']
                else:
                    data_to_decompress = compressed_data
                
                data_to_decompress = np.array(data_to_decompress).reshape(1, -1)
                decompressed = self.pca_models[level].inverse_transform(data_to_decompress)[0]
                return torch.tensor(decompressed, dtype=torch.float32)
            except Exception as e:
                warnings.warn(f"Error decompressing with PCA: {e}, using fallback")
                # Fallback to identity return if decompression fails
                if isinstance(compressed_data, dict) and 'data' in compressed_data:
                    return torch.tensor(compressed_data['data'], dtype=torch.float32)
                return torch.tensor(compressed_data, dtype=torch.float32)
    
    def _quantize(self, embedding: torch.Tensor, bits: int) -> np.ndarray:
        """Quantize a torch tensor to specified bit precision."""
        tensor = embedding.detach().cpu()
        
        # Get min and max for normalization
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Normalize to [0, 1]
        normalized = (tensor - min_val) / (max_val - min_val) if max_val > min_val else tensor
        
        # Quantize to specified bits
        levels = 2**bits - 1
        quantized = np.round(normalized.numpy() * levels).astype(np.uint8)
        
        # Store quantization parameters with the data
        result = {
            'data': quantized,
            'min_val': min_val,
            'max_val': max_val,
            'bits': bits
        }
        
        return result
    
    def _dequantize(self, quantized: Dict, bits: int) -> torch.Tensor:
        """Dequantize back to full precision."""
        levels = 2**bits - 1
        min_val = quantized['min_val']
        max_val = quantized['max_val']
        
        # Dequantize
        normalized = quantized['data'].astype(np.float32) / levels
        
        # Denormalize
        denormalized = normalized * (max_val - min_val) + min_val
        
        return torch.tensor(denormalized, dtype=torch.float32)
    
    def _quantize_array(self, arr: np.ndarray, bits: int) -> Dict:
        """Quantize a numpy array to specified bit precision."""
        # Get min and max for normalization
        min_val = arr.min()
        max_val = arr.max()
        
        # Normalize to [0, 1]
        normalized = (arr - min_val) / (max_val - min_val) if max_val > min_val else arr
        
        # Quantize to specified bits
        levels = 2**bits - 1
        quantized = np.round(normalized * levels).astype(np.uint8)
        
        # Store quantization parameters with the data
        result = {
            'data': quantized,
            'min_val': min_val,
            'max_val': max_val,
            'bits': bits
        }
        
        return result

class LifetimeHierarchicalStorage:
    """
    Hierarchical storage system designed to last a human lifetime (100+ years).
    Implements a tiered approach with different storage strategies for different time periods.
    """
    def __init__(self, 
                 base_path: str = "model_save/lifetime_memory", 
                 max_active_memories: int = 10_000_000,
                 total_capacity: int = 1_000_000_000_000,  # 1 trillion memories capacity 
                 embedding_dim: int = 768,
                 use_compression: bool = True,
                 compression_schedule: Dict[str, str] = None):
        """
        Initialize the lifetime hierarchical storage system.
        
        Args:
            base_path: Base path for storage
            max_active_memories: Maximum number of active memories
            total_capacity: Total capacity (active + all archives)
            embedding_dim: Dimension of memory embeddings
            use_compression: Whether to use compression for archived memories
            compression_schedule: Dictionary mapping age to compression level
                                 (e.g., {'1y': 'light', '5y': 'medium', '20y': 'heavy'})
        """
        self.base_path = base_path
        self.max_active_memories = max_active_memories
        self.total_capacity = total_capacity
        self.embedding_dim = embedding_dim
        self.use_compression = use_compression
        
        # Default compression schedule based on memory age
        self.compression_schedule = compression_schedule or {
            '0d': 'none',    # New memories: no compression
            '30d': 'light',  # 30 days old: light compression
            '1y': 'medium',  # 1 year old: medium compression
            '5y': 'heavy',   # 5 years old: heavy compression
            '20y': 'extreme' # 20 years old: extreme compression
        }
        
        # Initialize compression service if using compression
        if use_compression:
            self.compression_service = MemoryCompressionService(embedding_dim=embedding_dim)
        
        # Set up directory structure
        self._init_directory_structure()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Track current storage amount
        self.current_storage = 0
    
    def _init_directory_structure(self):
        """Initialize the directory structure for hierarchical storage."""
        # Create base path if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
        # Create active memories directory
        self.active_path = os.path.join(self.base_path, "active")
        os.makedirs(self.active_path, exist_ok=True)
        
        # Create archive directories for different time periods
        # Structure: /archive/[year]/[month]/
        # This allows efficient storage and retrieval by time period
        self.archive_path = os.path.join(self.base_path, "archive")
        os.makedirs(self.archive_path, exist_ok=True)
        
        # Create index directory for metadata and FAISS indexes
        self.index_path = os.path.join(self.base_path, "index")
        os.makedirs(self.index_path, exist_ok=True)
        
        # Create directory for global indexes
        self.global_index_path = os.path.join(self.index_path, "global")
        os.makedirs(self.global_index_path, exist_ok=True)
    
    def get_archive_path_for_time(self, timestamp: float) -> str:
        """
        Get the archive path for a specific time.
        
        Args:
            timestamp: UNIX timestamp
            
        Returns:
            Path for the archived memory
        """
        dt = datetime.datetime.fromtimestamp(timestamp)
        year_dir = os.path.join(self.archive_path, str(dt.year))
        month_dir = os.path.join(year_dir, f"{dt.month:02d}")
        
        # Create directories if they don't exist
        os.makedirs(month_dir, exist_ok=True)
        
        return month_dir
    
    def get_time_period_string(self, timestamp: float) -> str:
        """
        Get a string representation of the time period for a timestamp.
        
        Args:
            timestamp: UNIX timestamp
            
        Returns:
            String in format "YYYY_MM"
        """
        dt = datetime.datetime.fromtimestamp(timestamp)
        return f"{dt.year}_{dt.month:02d}"
    
    def get_compression_level_for_age(self, timestamp: float) -> str:
        """
        Determine the appropriate compression level based on memory age.
        
        Args:
            timestamp: UNIX timestamp of the memory
            
        Returns:
            Compression level
        """
        if not self.use_compression:
            return 'none'
        
        age_days = (time.time() - timestamp) / (24 * 3600)
        
        # Convert schedule keys to days
        schedule_days = {}
        for key, value in self.compression_schedule.items():
            if key.endswith('d'):
                schedule_days[int(key[:-1])] = value
            elif key.endswith('y'):
                schedule_days[int(key[:-1]) * 365] = value
            else:
                try:
                    schedule_days[int(key)] = value
                except ValueError:
                    pass
        
        # Sort by age (descending)
        sorted_ages = sorted(schedule_days.keys(), reverse=True)
        
        # Find appropriate compression level
        for age in sorted_ages:
            if age_days >= age:
                return schedule_days[age]
        
        # Default to no compression for very recent memories
        return 'none'
    
    def store_memory(self, memory_id: str, embedding: torch.Tensor, metadata: Dict, timestamp: float) -> str:
        """
        Store a memory in the appropriate location based on its characteristics.
        
        Args:
            memory_id: Unique ID for the memory
            embedding: Memory embedding tensor
            metadata: Memory metadata
            timestamp: Creation timestamp
            
        Returns:
            Storage location identifier
        """
        with self.lock:
            # Determine where to store based on current system state
            # For new memories, they go to active storage by default
            storage_path = self.active_path
            
            # Create a unique filename for this memory
            embedding_filename = f"{memory_id}.pt"
            metadata_filename = f"{memory_id}.json"
            
            # Get compression level (none for active memory)
            compression_level = 'none'
            
            # Store the embedding
            embedding_path = os.path.join(storage_path, embedding_filename)
            
            if compression_level == 'none':
                # Store without compression
                torch.save(embedding, embedding_path)
            else:
                # Compress and store
                compressed = self.compression_service.compress(embedding, level=compression_level)
                with open(embedding_path, 'wb') as f:
                    f.write(compressed)
            
            # Store metadata
            metadata_path = os.path.join(storage_path, metadata_filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Update current storage
            self.current_storage += 1
            
            return os.path.join(storage_path, memory_id)
    
    def retrieve_memory(self, memory_id: str, location: str = None) -> Tuple[torch.Tensor, Dict]:
        """
        Retrieve a memory by ID and optional location hint.
        
        Args:
            memory_id: Memory ID to retrieve
            location: Optional location hint (active, archive path, etc.)
            
        Returns:
            Tuple of (embedding tensor, metadata dict)
        """
        with self.lock:
            # If location is provided, check there first
            if location:
                embedding_path = os.path.join(location, f"{memory_id}.pt")
                metadata_path = os.path.join(location, f"{memory_id}.json")
                
                if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                    return self._load_memory(embedding_path, metadata_path)
            
            # Check in active memory
            embedding_path = os.path.join(self.active_path, f"{memory_id}.pt")
            metadata_path = os.path.join(self.active_path, f"{memory_id}.json")
            
            if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                return self._load_memory(embedding_path, metadata_path)
            
            # Check in archive
            # This is inefficient for large archives, but we'll typically have location hints
            # For production use, we'd maintain an index of memory ID to location
            for year_dir in glob.glob(os.path.join(self.archive_path, "*")):
                for month_dir in glob.glob(os.path.join(year_dir, "*")):
                    embedding_path = os.path.join(month_dir, f"{memory_id}.pt")
                    metadata_path = os.path.join(month_dir, f"{memory_id}.json")
                    
                    if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                        return self._load_memory(embedding_path, metadata_path)
            
            # Memory not found
            raise ValueError(f"Memory with ID {memory_id} not found")
    
    def _load_memory(self, embedding_path: str, metadata_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Load a memory from disk.
        
        Args:
            embedding_path: Path to the embedding file
            metadata_path: Path to the metadata file
            
        Returns:
            Tuple of (embedding tensor, metadata dict)
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if file is compressed
        is_compressed = metadata.get('storage', {}).get('compressed', False)
        compression_level = metadata.get('storage', {}).get('compression_level', 'none')
        
        # Load embedding
        if not is_compressed or compression_level == 'none':
            # Regular torch load
            embedding = torch.load(embedding_path)
        else:
            # Load compressed data
            with open(embedding_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress
            embedding = self.compression_service.decompress(compressed_data, level=compression_level)
        
        return embedding, metadata
    
    def archive_memory(self, memory_id: str, embedding: torch.Tensor, metadata: Dict, timestamp: float) -> str:
        """
        Move a memory from active to archive storage.
        
        Args:
            memory_id: Memory ID to archive
            embedding: Memory embedding
            metadata: Memory metadata
            timestamp: Creation timestamp
            
        Returns:
            New location of the archived memory
        """
        with self.lock:
            # Get archive path for this time
            archive_dir = self.get_archive_path_for_time(timestamp)
            
            # Get appropriate compression level
            compression_level = self.get_compression_level_for_age(timestamp)
            
            # Update metadata to include storage information
            if 'storage' not in metadata:
                metadata['storage'] = {}
            
            metadata['storage']['archived_date'] = time.time()
            metadata['storage']['compressed'] = compression_level != 'none'
            metadata['storage']['compression_level'] = compression_level
            metadata['storage']['time_period'] = self.get_time_period_string(timestamp)
            
            # Create filenames
            embedding_filename = f"{memory_id}.pt"
            metadata_filename = f"{memory_id}.json"
            
            # Save to archive
            embedding_path = os.path.join(archive_dir, embedding_filename)
            metadata_path = os.path.join(archive_dir, metadata_filename)
            
            # Save embedding with compression if needed
            if compression_level == 'none':
                # Store without compression
                torch.save(embedding, embedding_path)
            else:
                # Compress and store, passing memory_id for TOVA pattern persistence
                compressed = self.compression_service.compress(
                    embedding, 
                    level=compression_level,
                    memory_id=memory_id  # Pass memory_id for TOVA persistence
                )
                with open(embedding_path, 'wb') as f:
                    f.write(compressed)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Remove from active storage if it exists there
            active_embedding_path = os.path.join(self.active_path, embedding_filename)
            active_metadata_path = os.path.join(self.active_path, metadata_filename)
            
            if os.path.exists(active_embedding_path):
                os.remove(active_embedding_path)
            
            if os.path.exists(active_metadata_path):
                os.remove(active_metadata_path)
            
            return os.path.join(archive_dir, memory_id)
    
    def recompress_memory(self, memory_id: str, location: str, new_level: str) -> str:
        """
        Recompress a memory with a different compression level.
        
        Args:
            memory_id: Memory ID
            location: Current location
            new_level: New compression level
            
        Returns:
            Updated location string
        """
        if not self.use_compression:
            return location
        
        with self.lock:
            # Load the memory
            embedding, metadata = self.retrieve_memory(memory_id, location)
            
            # Update storage information
            if 'storage' not in metadata:
                metadata['storage'] = {}
            
            metadata['storage']['compressed'] = new_level != 'none'
            metadata['storage']['compression_level'] = new_level
            metadata['storage']['recompressed_date'] = time.time()
            
            # Determine directory (use existing location)
            directory = os.path.dirname(location)
            
            # Create filenames
            embedding_filename = f"{memory_id}.pt"
            metadata_filename = f"{memory_id}.json"
            
            # Save with new compression
            embedding_path = os.path.join(directory, embedding_filename)
            metadata_path = os.path.join(directory, metadata_filename)
            
            # Compress and save embedding
            if new_level == 'none':
                torch.save(embedding, embedding_path)
            else:
                # Pass memory_id for TOVA pattern persistence
                compressed = self.compression_service.compress(
                    embedding, 
                    level=new_level,
                    memory_id=memory_id
                )
                with open(embedding_path, 'wb') as f:
                    f.write(compressed)
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            return location
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage usage.
        
        Returns:
            Dictionary of storage statistics
        """
        with self.lock:
            # Count active memories
            active_count = len(glob.glob(os.path.join(self.active_path, "*.pt")))
            
            # Count archive memories by year/month
            archive_counts = {}
            total_archive_count = 0
            
            for year_dir in glob.glob(os.path.join(self.archive_path, "*")):
                year = os.path.basename(year_dir)
                archive_counts[year] = {}
                
                for month_dir in glob.glob(os.path.join(year_dir, "*")):
                    month = os.path.basename(month_dir)
                    count = len(glob.glob(os.path.join(month_dir, "*.pt")))
                    archive_counts[year][month] = count
                    total_archive_count += count
            
            # Calculate approximate storage usage
            # This is a rough estimate based on embedding size and compression
            bytes_per_embedding = self.embedding_dim * 4  # 4 bytes per float32
            
            # Estimate compressed sizes based on our compression schedule
            compressed_ratios = {
                'none': 1.0,
                'light': 0.5,  # 16-bit instead of 32-bit
                'medium': 0.25,  # ~25% of original size after PCA and precision reduction
                'heavy': 0.1,  # ~10% after PCA and 8-bit quantization
                'extreme': 0.05  # ~5% after extreme compression
            }
            
            # Calculate storage in bytes
            active_storage = active_count * bytes_per_embedding
            
            # For archive, we'd need to check each file's compression level
            # For this example, we'll just use a rough estimate based on age
            archive_storage = 0
            current_year = datetime.datetime.now().year
            current_month = datetime.datetime.now().month
            
            for year_str, months in archive_counts.items():
                year = int(year_str)
                for month_str, count in months.items():
                    month = int(month_str)
                    
                    # Calculate age in years
                    age_years = (current_year - year) + (current_month - month) / 12
                    
                    # Determine compression level based on age
                    if age_years < 1:
                        level = 'light'
                    elif age_years < 5:
                        level = 'medium'
                    elif age_years < 20:
                        level = 'heavy'
                    else:
                        level = 'extreme'
                    
                    # Calculate storage for this month
                    ratio = compressed_ratios[level]
                    month_storage = count * bytes_per_embedding * ratio
                    archive_storage += month_storage
            
            total_storage = active_storage + archive_storage
            
            return {
                'active_memories': active_count,
                'archive_memories': total_archive_count,
                'total_memories': active_count + total_archive_count,
                'active_storage_bytes': active_storage,
                'archive_storage_bytes': archive_storage,
                'total_storage_bytes': total_storage,
                'storage_usage_percent': (total_storage / (self.total_capacity * bytes_per_embedding)) * 100,
                'archive_breakdown': archive_counts,
                'compression_enabled': self.use_compression
            }

class EpisodicMemory(nn.Module):
    """
    Ultra-high capacity episodic memory system based on the Titans architecture.
    Implements deep neural memory with multiple integration options and hierarchical storage
    for effectively unlimited memory across human timescales (100+ years).
    """
    def __init__(self, 
                 embedding_dim: int = 768, 
                 hidden_dim: int = 512,
                 capacity: int = 10_000_000_000,  # 10 billion memories (capacity for a lifetime)
                 max_active_memories: int = 10_000_000,  # 10 million active memories
                 surprise_threshold: float = 0.7,
                 config_path: str = "model_save/episodic_memory_config.json",
                 use_faiss: bool = True,  # Default to True for efficient similarity search
                 num_neural_layers: int = 3,
                 persistent_items: int = 64,  # Increased for better task knowledge
                 integration_type: str = "MAC",
                 num_attention_heads: int = 8,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 forget_rate: float = 0.05,  # Reduced forget rate for lifetime memory
                 archive_path: str = "model_save/memory_archive",
                 memory_compression_factor: float = 0.8,  # Compression factor for archived memories
                 use_lifetime_storage: bool = True,
                 multithread_operations: bool = True,
                 memory_retention_policy: str = "importance_based"):
        """
        Initialize the ultra-high capacity episodic memory system.
        
        Args:
            embedding_dim: Dimension of memory embeddings
            hidden_dim: Hidden dimension for the neural memory
            capacity: Maximum number of memories to store in total (active + archived)
            max_active_memories: Maximum number of memories to keep in active memory
            surprise_threshold: Threshold for determining if something is surprising
            config_path: Path to save configuration including agent information
            use_faiss: Whether to use FAISS for similarity search
            num_neural_layers: Number of layers in neural memory
            persistent_items: Number of persistent memory items
            integration_type: Type of memory integration ('MAC', 'MAG', or 'MAL')
            num_attention_heads: Number of attention heads for memory integration
            learning_rate: Learning rate for neural memory updates
            momentum: Momentum factor for surprise tracking
            forget_rate: Base rate for adaptive forgetting
            archive_path: Path for storing archived memories
            memory_compression_factor: Compression factor for archived memories (0.0-1.0)
            use_lifetime_storage: Whether to use lifetime storage features
            multithread_operations: Use multithreading for non-critical operations
            memory_retention_policy: Policy for deciding which memories to retain
                                    ('importance_based', 'recency_based', or 'hybrid')
        """
        super(EpisodicMemory, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.max_active_memories = max_active_memories
        self.surprise_threshold = surprise_threshold
        self.config_path = config_path
        self.use_faiss = use_faiss
        self.archive_path = archive_path
        self.memory_compression_factor = memory_compression_factor
        self.use_lifetime_storage = use_lifetime_storage
        self.multithread_operations = multithread_operations
        self.memory_retention_policy = memory_retention_policy
        
        # Neural memory module (as described in Titans)
        self.neural_memory = NeuralMemoryModule(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_neural_layers,
            surprise_threshold=surprise_threshold,
            learning_rate=learning_rate,
            momentum=momentum,
            forget_rate=forget_rate
        )
        
        # Persistent memory module (input-independent)
        self.persistent_memory = PersistentMemory(
            embedding_dim=embedding_dim,
            num_items=persistent_items
        )
        
        # Memory integration module
        self.memory_integration = MemoryIntegration(
            embedding_dim=embedding_dim,
            integration_type=integration_type,
            num_heads=num_attention_heads
        )
        
        # Primary storage parameters
        self.memories: List[MemoryItem] = []
        self.memory_embeddings = torch.zeros((0, embedding_dim), dtype=torch.float32)
        
        # Create hierarchical storage system for lifetime memory
        base_storage_path = os.path.dirname(archive_path)
        os.makedirs(base_storage_path, exist_ok=True)
        
        self.storage = LifetimeHierarchicalStorage(
            base_path=base_storage_path,
            max_active_memories=max_active_memories,
            total_capacity=capacity,
            embedding_dim=embedding_dim,
            use_compression=True,  # Enable compression for archived memories
            neural_memory=self.neural_memory  # Pass neural memory for persistent TOVA patterns
        )
        
        # Archive storage for long-term persistence
        os.makedirs(self.archive_path, exist_ok=True)
        self.archived_memories = {}  # Maps archive_id to metadata
        self.archive_index = None
        self._init_archive_storage()
        
        # FAISS index for efficient similarity search of active memories
        if use_faiss:
            # Use hierarchical FAISS index with IVF for billion-scale search
            self.index = self._create_billion_scale_index(embedding_dim)
        else:
            self.index = None
        
        # Memory cache for fast retrieval of frequently accessed memories
        self.cache = LRUCache(min(max_active_memories // 10, 10000))  # 10% of active capacity or max 10k
        
        # Track known agents
        self.known_agents = {}
        
        # Statistics for lifetime memory management
        self.stats = LifetimeMemoryStats()
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if multithread_operations else None
        
        # Background task management
        self.background_tasks = []
        
        # Importance scoring parameters for memory retention
        self.importance_alpha = 0.5  # Weight for surprise level (0-1)
        self.importance_beta = 0.3   # Weight for access count (0-1)
        self.importance_gamma = 0.2  # Weight for recency (0-1)
        
        # Lock for thread safety on memory operations
        self.memory_lock = threading.RLock()
        
        # Register automatic management functions
        if use_lifetime_storage:
            self._schedule_maintenance()
            
        # Load existing configuration if available
        self._load_config()
    
    def _create_billion_scale_index(self, embedding_dim):
        """
        Create a FAISS index optimized for billion-scale vector search.
        
        Args:
            embedding_dim: Dimension of the embeddings
            
        Returns:
            FAISS index configured for billion-scale search
        """
        # For billion-scale search, we use IVF (Inverted File Index) with flat quantization
        # The number of centroids (nlist) should scale with sqrt(n) where n is the number of vectors
        # For a billion vectors, ~30K-100K centroids is reasonable
        nlist = 50000  # Number of centroids for billion-scale
        
        # Create a quantizer (the index that computes centroids)
        quantizer = faiss.IndexFlatL2(embedding_dim)
        
        # Create the full index: IVF with flat storage
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        
        # To improve search speed at the cost of some accuracy, we can set nprobe
        # nprobe is the number of nearest centroids to search
        index.nprobe = 256
        
        # Initialize with an empty training set (will be trained incrementally)
        # Note: For production, we would want to train this on a representative dataset
        index.is_trained = False
        
        return index
    
    def _init_archive_storage(self):
        """Initialize archive storage system and load any existing archived memories index."""
        # Create archive directory if it doesn't exist
        if not os.path.exists(self.archive_path):
            os.makedirs(self.archive_path, exist_ok=True)
            
        # Create index directory
        index_dir = os.path.join(self.archive_path, "index")
        os.makedirs(index_dir, exist_ok=True)
        
        # Create archive index file path
        self.archive_index_path = os.path.join(index_dir, "archive_index.json")
        
        # Load archive index if it exists
        if os.path.exists(self.archive_index_path):
            try:
                with open(self.archive_index_path, 'r') as f:
                    self.archived_memories = json.load(f)
                self.stats.archive_memory_count = len(self.archived_memories)
                print(f"Loaded archive index with {len(self.archived_memories)} archived memories.")
            except Exception as e:
                print(f"Error loading archive index: {e}")
                self.archived_memories = {}
        else:
            self.archived_memories = {}
        
        # Create archive FAISS indexes for different time periods
        self._init_archive_faiss_indexes()
    
    def _init_archive_faiss_indexes(self):
        """Initialize FAISS indexes for archived memories by time periods."""
        self.archive_indexes = {}  # Maps time period (e.g., "2023_01") to FAISS index
        
        # Get list of existing time periods in the archive
        time_periods = set()
        for memory_meta in self.archived_memories.values():
            time_period = memory_meta.get('time_period')
            if time_period:
                time_periods.add(time_period)
        
        # Create or load FAISS index for each time period
        for time_period in time_periods:
            self._get_archive_index_for_period(time_period)
    
    def _get_archive_index_for_period(self, time_period):
        """
        Get or create FAISS index for a specific time period.
        
        Args:
            time_period: Time period string (e.g., "2023_01" for Jan 2023)
            
        Returns:
            FAISS index for the specified time period
        """
        if time_period in self.archive_indexes:
            return self.archive_indexes[time_period]
        
        # Create directory for this time period if it doesn't exist
        period_dir = os.path.join(self.archive_path, time_period)
        os.makedirs(period_dir, exist_ok=True)
        
        # Path for FAISS index
        index_path = os.path.join(period_dir, "faiss_index.bin")
        
        # Check if index already exists
        if os.path.exists(index_path):
            try:
                # Load existing index
                self.archive_indexes[time_period] = faiss.read_index(index_path)
                return self.archive_indexes[time_period]
            except Exception as e:
                print(f"Error loading archive index for {time_period}: {e}")
        
        # Create new index for this time period
        index = self._create_billion_scale_index(self.embedding_dim)
        self.archive_indexes[time_period] = index
        
        # Try to load embeddings for memories in this time period
        memory_ids = [memory_id for memory_id, meta in self.archived_memories.items() 
                     if meta.get('time_period') == time_period]
        
        if memory_ids:
            # Load embeddings for these memories
            for memory_id in memory_ids:
                embedding_path = os.path.join(period_dir, f"{memory_id}.pt")
                if os.path.exists(embedding_path):
                    try:
                        embedding = torch.load(embedding_path).detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                        
                        # If the index isn't trained yet, train it
                        if not index.is_trained:
                            index.train(embedding)
                        
                        # Add to index
                        index.add(embedding)
                    except Exception as e:
                        print(f"Error loading embedding for {memory_id}: {e}")
        
        # Save the index if memories were added
        if index.ntotal > 0:
            faiss.write_index(index, index_path)
        
        return index
    
    def _load_config(self):
        """Load configuration from disk if available."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    if 'known_agents' in config:
                        self.known_agents = config['known_agents']
                    print(f"Loaded episodic memory configuration with {len(self.known_agents)} known agents.")
            except Exception as e:
                print(f"Error loading episodic memory configuration: {e}")
    
    def _save_config(self):
        """Save configuration to disk."""
        config_dir = os.path.dirname(self.config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        config = {
            'known_agents': self.known_agents,
            'embedding_dim': self.embedding_dim,
            'capacity': self.capacity,
            'max_active_memories': self.max_active_memories,
            'surprise_threshold': self.surprise_threshold,
            'memory_count': len(self.memories),
            'archive_count': len(self.archived_memories),
            'last_updated': datetime.datetime.now().isoformat(),
            'stats': self.stats.to_dict()
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f)
            print(f"Episodic memory configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving episodic memory configuration: {e}")
    
    def save_state(self, path: str):
        """
        Save the state of the episodic memory.
        
        Args:
            path: Path to save the state to
        """
        state_dir = os.path.dirname(path)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            
        # Create a simplified representation of memories
        memory_data = []
        for memory in self.memories:
            memory_data.append([
                memory.embedding.tolist(),
                memory.timestamp,
                memory.surprise_level,
                memory.agent_info_id,
                memory.metadata
            ])
        
        state = {
            'embedding_dim': self.embedding_dim,
            'capacity': self.capacity,
            'surprise_threshold': self.surprise_threshold,
            'known_agents': self.known_agents,
            'memories': memory_data,
            'archived_count': len(self.archived_memories),
            'last_updated': datetime.datetime.now().isoformat(),
            'stats': self.stats.to_dict()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(state, f)
            print(f"Episodic memory state saved to {path}")
        except Exception as e:
            print(f"Error saving episodic memory state: {e}")
    
    def load_state(self, path: str):
        """
        Load the state of the episodic memory.
        
        Args:
            path: Path to load the state from
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.embedding_dim = state['embedding_dim']
            self.capacity = state['capacity']
            self.surprise_threshold = state['surprise_threshold']
            self.known_agents = state['known_agents']
            
            # Rebuild memories
            self.memories = []
            for emb, ts, sl, aid, meta in state['memories']:
                embedding = torch.tensor(emb, dtype=torch.float32)
                memory = MemoryItem(
                    embedding=embedding,
                    timestamp=ts,
                    surprise_level=sl,
                    agent_info_id=aid,
                    metadata=meta
                )
                self.memories.append(memory)
            
            # Update memory embeddings and index
            self._update_memory_embeddings()
            
            if self.use_faiss and len(self.memories) > 0:
                # Check if we need to recreate the billion-scale index
                if isinstance(self.index, faiss.IndexFlatL2):
                    self.index = self._create_billion_scale_index(self.embedding_dim)
                
                # Convert embeddings to numpy
                embeddings_np = self.memory_embeddings.detach().cpu().numpy().astype(np.float32)
                
                # Train index if needed (for IVF indexes)
                if not self.index.is_trained and isinstance(self.index, faiss.IndexIVFFlat):
                    self.index.train(embeddings_np)
                
                # Add embeddings to index
                self.index.add(embeddings_np)
            
            print(f"Episodic memory state loaded from {path} with {len(self.memories)} memories")
        except Exception as e:
            print(f"Error loading episodic memory state: {e}")
    
    def process_with_neural_memory(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process input with neural memory only.
        
        Args:
            input_tensor: Input tensor to process
            
        Returns:
            Processed tensor from neural memory
        """
        # Ensure input is on the correct device
        device = next(self.neural_memory.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Get neural memory output
        return self.neural_memory(input_tensor)
    
    def get_persistent_memory(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get persistent memory items.
        
        Args:
            batch_size: Batch size to expand for
            
        Returns:
            Persistent memory tensor
        """
        return self.persistent_memory(batch_size)
    
    def integrate_memory(self, 
                        input_tensor: torch.Tensor, 
                        memory_context: torch.Tensor) -> torch.Tensor:
        """
        Integrate input with memory context using the configured integration method.
        
        Args:
            input_tensor: Input tensor
            memory_context: Memory context tensor
            
        Returns:
            Integrated tensor
        """
        return self.memory_integration(input_tensor, memory_context)
    
    def parallel_process_chunks(self, input_chunks: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process multiple input chunks in parallel using the neural memory.
        Implements the parallelization approach described in the Titans paper.
        
        Args:
            input_chunks: List of input tensor chunks
            
        Returns:
            List of processed output chunks
        """
        # Stack chunks for batch processing
        batch = torch.cat([chunk.unsqueeze(0) for chunk in input_chunks], dim=0)
        
        # Get device
        device = next(self.neural_memory.parameters()).device
        batch = batch.to(device)
        
        # Process in batch
        outputs = self.neural_memory(batch)
        
        # Split back into individual chunks
        return [outputs[i] for i in range(outputs.size(0))]
    
    def _update_memory_embeddings(self):
        """Update the tensor of memory embeddings from the list of memories."""
        if len(self.memories) == 0:
            self.memory_embeddings = torch.zeros((0, self.embedding_dim), dtype=torch.float32)
            return
        
        # Stack embeddings from all memories
        self.memory_embeddings = torch.stack([memory.embedding for memory in self.memories])
    
    def _calculate_memory_importance(self, memory: MemoryItem) -> float:
        """
        Calculate the importance score for a memory item.
        
        Args:
            memory: Memory item to score
            
        Returns:
            Importance score (higher means more important)
        """
        # Factor 1: Surprise level
        surprise_score = memory.surprise_level
        
        # Factor 2: Access frequency (log scale)
        # Memories accessed more often are more important
        access_score = math.log1p(memory.access_count) / 10.0  # Normalize
        
        # Factor 3: Recency (inverse of time since last access)
        # More recently accessed memories are more important
        time_since_access = max(0.001, time.time() - memory.last_accessed)  # Avoid division by zero
        recency_score = 1.0 / (1.0 + math.log1p(time_since_access / (24 * 3600)))  # Normalize to 0-1 with day units
        
        # Combine factors with weights
        importance = (
            self.importance_alpha * surprise_score +
            self.importance_beta * access_score +
            self.importance_gamma * recency_score
        )
        
        return importance
    
    def add_memory(self, 
                  embedding: torch.Tensor, 
                  surprise_level: float = None, 
                  agent_info_id: str = None,
                  metadata: Dict[str, Any] = None,
                  timestamp: float = None) -> str:
        """
        Add a new memory to the episodic memory.
        
        Args:
            embedding: The embedding tensor
            surprise_level: Optional surprise level (if None, calculated using neural memory)
            agent_info_id: Optional agent ID associated with this memory
            metadata: Optional metadata for the memory
            timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            ID of the created memory
        """
        with self.memory_lock:
            # Ensure embedding is a tensor
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            # Normalize embedding for consistent similarity calculation
            embedding = F.normalize(embedding, p=2, dim=-1)
            
            # Calculate surprise level if not provided
            if surprise_level is None:
                # Use neural memory to calculate surprise
                with torch.no_grad():
                    device = next(self.neural_memory.parameters()).device
                    embedding_device = embedding.to(device)
                    surprise_tensor = self.neural_memory.calculate_surprise(embedding_device.unsqueeze(0), embedding_device.unsqueeze(0))
                    surprise_level = surprise_tensor.item()
            
            # Create memory item
            memory = MemoryItem(
                embedding=embedding,
                timestamp=timestamp,
                surprise_level=surprise_level,
                agent_info_id=agent_info_id,
                metadata=metadata or {}
            )
            
            # Add to active memory
            self.memories.append(memory)
            self.stats.record_memory_added()
            
            # Update memory embeddings tensor
            self._update_memory_embeddings()
            
            # Update FAISS index if using
            if self.use_faiss:
                embedding_np = embedding.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                
                # For IVF indexes, we need to train first if it's not trained
                if not self.index.is_trained and isinstance(self.index, faiss.IndexIVFFlat):
                    self.index.train(embedding_np)
                
                self.index.add(embedding_np)
            
            # Also store in hierarchical storage for persistence
            if self.use_lifetime_storage:
                location = self.storage.store_memory(
                    memory_id=memory.id, 
                    embedding=embedding, 
                    metadata=memory.to_dict(), 
                    timestamp=memory.timestamp
                )
                
                # Add storage location to metadata
                memory.metadata['storage_location'] = location
            
            # Check if we need to archive old memories
            if len(self.memories) > self.max_active_memories:
                self._archive_old_memories()
            
            return memory.id
    
    def _archive_old_memories(self, count: int = None):
        """
        Move old memories from active to archive storage.
        
        Args:
            count: Number of memories to archive (if None, archives to reach max_active_memories)
        """
        with self.memory_lock:
            if len(self.memories) <= self.max_active_memories and count is None:
                return
            
            # Determine how many memories to archive
            num_to_archive = count
            if num_to_archive is None:
                num_to_archive = len(self.memories) - self.max_active_memories
                num_to_archive = max(1, min(num_to_archive, len(self.memories) // 10))  # Archive at most 10% at once
            
            # Calculate importance for all memories
            memory_with_importance = [(memory, self._calculate_memory_importance(memory)) 
                                     for memory in self.memories]
            
            # Sort by importance (ascending, so least important first)
            memory_with_importance.sort(key=lambda x: x[1])
            
            # Get memories to archive
            memories_to_archive = [mem for mem, _ in memory_with_importance[:num_to_archive]]
            
            # Archive each memory
            for memory in memories_to_archive:
                # Archive in hierarchical storage
                if self.use_lifetime_storage:
                    time_period = self.storage.get_time_period_string(memory.timestamp)
                    
                    location = self.storage.archive_memory(
                        memory_id=memory.id,
                        embedding=memory.embedding,
                        metadata=memory.to_dict(),
                        timestamp=memory.timestamp
                    )
                    
                    # Record in archived_memories index
                    self.archived_memories[memory.id] = {
                        'time_period': time_period,
                        'timestamp': memory.timestamp,
                        'surprise_level': memory.surprise_level,
                        'agent_info_id': memory.agent_info_id,
                        'last_accessed': memory.last_accessed,
                        'storage_location': location
                    }
                    
                    # Update FAISS index for this time period if exists
                    if time_period in self.archive_indexes:
                        embedding_np = memory.embedding.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                        
                        # Train if needed
                        if not self.archive_indexes[time_period].is_trained and isinstance(self.archive_indexes[time_period], faiss.IndexIVFFlat):
                            self.archive_indexes[time_period].train(embedding_np)
                        
                        # Add to index
                        self.archive_indexes[time_period].add(embedding_np)
                
                # Update stats
                self.stats.record_memory_archived(self.storage.get_time_period_string(memory.timestamp))
            
            # Remove archived memories from active list
            archived_ids = {memory.id for memory in memories_to_archive}
            self.memories = [memory for memory in self.memories if memory.id not in archived_ids]
            
            # Update memory embeddings and index
            self._update_memory_embeddings()
            
            if self.use_faiss:
                # For billion-scale usage, we can't rebuild the index every time
                # We would need to maintain a mapping of memory_id to index position
                # and use selective deletion
                if isinstance(self.index, faiss.IndexFlatL2):
                    # For flat index, we can just rebuild
                    self.index.reset()
                    if len(self.memories) > 0:
                        embeddings_np = self.memory_embeddings.detach().cpu().numpy().astype(np.float32)
                        self.index.add(embeddings_np)
                else:
                    # For billion-scale IVF index, rebuilding is very expensive
                    # We would need to track IDs and use selective delete if supported
                    # For now, we'll rebuild completely but in production would need optimization
                    self.index = self._create_billion_scale_index(self.embedding_dim)
                    if len(self.memories) > 0:
                        embeddings_np = self.memory_embeddings.detach().cpu().numpy().astype(np.float32)
                        if not self.index.is_trained:
                            self.index.train(embeddings_np)
                        self.index.add(embeddings_np)
            
            # Save archive index
            self._save_archive_index()
    
    def _save_archive_index(self):
        """Save the archive index to disk."""
        # Ensure index directory exists
        index_dir = os.path.join(self.archive_path, "index")
        os.makedirs(index_dir, exist_ok=True)
        
        # Save the archive index
        try:
            with open(self.archive_index_path, 'w') as f:
                json.dump(self.archived_memories, f)
        except Exception as e:
            print(f"Error saving archive index: {e}")
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory item or None if not found
        """
        with self.memory_lock:
            # Check cache first
            cached = self.cache.get(memory_id)
            if cached:
                cached.mark_accessed()
                return cached
            
            # Check active memories
            for memory in self.memories:
                if memory.id == memory_id:
                    memory.mark_accessed()
                    self.cache.put(memory_id, memory)
                    return memory
            
            # Check archived memories
            if memory_id in self.archived_memories:
                archive_meta = self.archived_memories[memory_id]
                
                # Retrieve from hierarchical storage
                if self.use_lifetime_storage and 'storage_location' in archive_meta:
                    location = archive_meta['storage_location']
                    try:
                        embedding, metadata = self.storage.retrieve_memory(memory_id, location)
                        
                        # Create memory item
                        memory = MemoryItem.from_dict(metadata, embedding)
                        memory.mark_accessed()
                        
                        # Add to cache
                        self.cache.put(memory_id, memory)
                        
                        # Update archive metadata
                        archive_meta['last_accessed'] = memory.last_accessed
                        self._save_archive_index()
                        
                        # Update stats
                        self.stats.record_memory_retrieved()
                        
                        return memory
                    except Exception as e:
                        print(f"Error retrieving archived memory {memory_id}: {e}")
            
            return None
    
    def retrieve_relevant_memories(self, 
                                  query_embedding: torch.Tensor, 
                                  top_k: int = 5,
                                  use_neural: bool = True,
                                  search_archive: bool = True,
                                  time_period: str = None) -> List[MemoryItem]:
        """
        Retrieve memories relevant to the query embedding.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Maximum number of memories to return
            use_neural: Whether to use neural memory for retrieval
            search_archive: Whether to search archived memories
            time_period: Optional time period to limit archive search
            
        Returns:
            List of relevant memory items
        """
        start_time = time.time()
        
        with self.memory_lock:
            # Ensure query is a tensor
            if not isinstance(query_embedding, torch.Tensor):
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            
            # Normalize for consistent similarity
            query_embedding = F.normalize(query_embedding.float(), p=2, dim=-1)
            
            results = []
            
            # Get relevant memories from active memory
            if len(self.memories) > 0:
                if self.use_faiss:
                    # Use FAISS for efficient similarity search
                    query_np = query_embedding.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                    distances, indices = self.index.search(query_np, min(top_k, len(self.memories)))
                    
                    # Get memory items from indices
                    active_results = [self.memories[idx] for idx in indices[0] if idx >= 0 and idx < len(self.memories)]
                else:
                    # Calculate similarity with all memories
                    similarities = torch.matmul(query_embedding, self.memory_embeddings.t()).squeeze()
                    
                    # Get top k indices
                    if len(similarities.shape) == 0:  # Single value
                        top_indices = [0] if similarities.item() > 0 else []
                    else:
                        top_values, top_indices = torch.topk(similarities, min(top_k, len(self.memories)))
                        # Filter out negative similarities
                        top_indices = [idx.item() for idx, val in zip(top_indices, top_values) if val.item() > 0]
                    
                    # Get memory items from indices
                    active_results = [self.memories[idx] for idx in top_indices]
                
                # Mark memories as accessed
                for memory in active_results:
                    memory.mark_accessed()
                
                results.extend(active_results)
            
            # Use neural memory if requested
            if use_neural:
                # Get neural memory output
                device = next(self.neural_memory.parameters()).device
                query_device = query_embedding.to(device)
                neural_output = self.neural_memory(query_device.unsqueeze(0)).squeeze(0)
                
                # Create a virtual memory item for the neural output
                neural_memory = MemoryItem(
                    embedding=neural_output.detach().cpu(),
                    timestamp=time.time(),
                    surprise_level=0.5,  # Middle surprise level
                    metadata={'source': 'neural_memory'}
                )
                
                results.append(neural_memory)
            
            # Search archive if requested and we still need more results
            if search_archive and len(results) < top_k:
                archive_results = self._search_archive(query_embedding, top_k - len(results), time_period)
                results.extend(archive_results)
            
            # Sort by relevance (dot product similarity)
            results.sort(key=lambda memory: torch.dot(query_embedding, memory.embedding).item(), reverse=True)
            
            return results
    
    def _search_archive(self, query_embedding: torch.Tensor, 
                       top_k: int, 
                       time_period: str = None) -> List[MemoryItem]:
        """
        Search archived memories for relevant items.
        
        Args:
            query_embedding: Query embedding
            top_k: Maximum number of memories to return
            time_period: Optional time period to limit search
            
        Returns:
            List of relevant archived memory items
        """
        results = []
        query_np = query_embedding.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
        
        # Determine which time periods to search
        if time_period:
            # Search just this time period
            time_periods = [time_period]
        else:
            # Search all time periods, or most recent if too many
            time_periods = list(self.archive_indexes.keys())
            if len(time_periods) > 10:
                # If more than 10 time periods, focus on most recent
                time_periods.sort(reverse=True)
                time_periods = time_periods[:10]
        
        # Search each time period
        for period in time_periods:
            if period in self.archive_indexes:
                index = self.archive_indexes[period]
                
                # Skip if this index is empty
                if index.ntotal == 0:
                    continue
                
                # Search this index
                try:
                    distances, indices = index.search(query_np, min(top_k, index.ntotal))
                    
                    # For each index, get the memory ID
                    # Note: This requires a mapping from FAISS index position to memory ID
                    # For this example, we'll assume sequential addition and use glob to get IDs
                    period_dir = os.path.join(self.archive_path, period)
                    memory_files = glob.glob(os.path.join(period_dir, "*.pt"))
                    memory_ids = [os.path.basename(f).split('.')[0] for f in memory_files]
                    
                    if indices.shape[1] > 0 and len(memory_ids) > 0:
                        # Get valid indices (some might be -1 if not enough results)
                        valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(memory_ids)]
                        
                        # Get memory IDs from indices
                        for idx in valid_indices:
                            if idx < len(memory_ids):
                                memory_id = memory_ids[idx]
                                memory = self.retrieve_memory(memory_id)
                                if memory:
                                    results.append(memory)
                except Exception as e:
                    print(f"Error searching archive index for period {period}: {e}")
        
        return results
    
    def query(self, 
             query_embedding: torch.Tensor, 
             top_k: int = 5,
             use_neural: bool = True,
             search_archive: bool = True,
             time_period: str = None,
             max_age_days: float = None) -> Tuple[List[MemoryItem], float]:
        """
        Perform a memory query with additional filtering options.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Maximum number of memories to return
            use_neural: Whether to use neural memory
            search_archive: Whether to search archived memories
            time_period: Optional time period to limit search
            max_age_days: Maximum age of memories to retrieve (in days)
            
        Returns:
            Tuple of (list of memory items, average relevance score)
        """
        # Get relevant memories
        memories = self.retrieve_relevant_memories(
            query_embedding, 
            top_k=top_k * 2,  # Get more than needed for filtering
            use_neural=use_neural,
            search_archive=search_archive,
            time_period=time_period
        )
        
        # Apply age filter if specified
        if max_age_days is not None:
            max_age_seconds = max_age_days * 24 * 3600
            now = time.time()
            memories = [mem for mem in memories if (now - mem.timestamp) <= max_age_seconds]
        
        # Calculate relevance scores
        scores = []
        for memory in memories:
            # Calculate dot product similarity
            similarity = torch.dot(query_embedding, memory.embedding).item()
            scores.append(similarity)
        
        # Sort by relevance
        memory_with_scores = list(zip(memories, scores))
        memory_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_memories = [mem for mem, _ in memory_with_scores[:top_k]]
        avg_relevance = sum(scores) / len(scores) if scores else 0
        
        return top_memories, avg_relevance
    
    def forget_memory(self, memory_id: str) -> bool:
        """
        Explicitly forget a memory by ID.
        
        Args:
            memory_id: ID of memory to forget
            
        Returns:
            True if memory was forgotten, False otherwise
        """
        with self.memory_lock:
            # Check if in active memories
            for i, memory in enumerate(self.memories):
                if memory.id == memory_id:
                    # Remove from active memories
                    del self.memories[i]
                    
                    # Update embeddings
                    self._update_memory_embeddings()
                    
                    # Update FAISS index
                    if self.use_faiss:
                        if isinstance(self.index, faiss.IndexFlatL2):
                            # Rebuild flat index
                            self.index.reset()
                            if len(self.memories) > 0:
                                embeddings_np = self.memory_embeddings.detach().cpu().numpy().astype(np.float32)
                                self.index.add(embeddings_np)
                    
                    # Remove from cache
                    _ = self.cache.get(memory_id)  # This removes it from the LRU cache
                    
                    return True
            
            # Check if in archive
            if memory_id in self.archived_memories:
                archive_meta = self.archived_memories[memory_id]
                
                # Remove from archive
                if self.use_lifetime_storage and 'storage_location' in archive_meta:
                    location = archive_meta['storage_location']
                    
                    # Delete files
                    location_dir = os.path.dirname(location)
                    embedding_path = os.path.join(location_dir, f"{memory_id}.pt")
                    metadata_path = os.path.join(location_dir, f"{memory_id}.json")
                    
                    if os.path.exists(embedding_path):
                        os.remove(embedding_path)
                    
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                
                # Remove from archive index
                del self.archived_memories[memory_id]
                self._save_archive_index()
                
                # Update stats
                self.stats.archive_memory_count -= 1
                
                # Remove from cache
                _ = self.cache.get(memory_id)  # This removes it from the LRU cache
                
                return True
            
            return False
    
    def _schedule_maintenance(self):
        """Schedule periodic maintenance tasks for the memory system."""
        # Define maintenance tasks
        def run_maintenance():
            try:
                # 1. Archive old memories if active memory is getting full
                active_capacity_threshold = self.max_active_memories * 0.9  # 90% of max
                if len(self.memories) > active_capacity_threshold:
                    self._archive_old_memories()
                
                # 2. Optimize FAISS indexes
                # In a real system, we would periodically rebuild/optimize the indexes
                
                # 3. Compress old memories based on age
                if self.use_lifetime_storage:
                    self._compress_old_memories()
                
                # 4. Save configuration
                self._save_config()
                
            except Exception as e:
                print(f"Error during memory maintenance: {e}")
            
            # Schedule next maintenance
            # In a real system, we'd use a proper scheduler
            if self.multithread_operations and self.thread_pool:
                self.background_tasks.append(
                    self.thread_pool.submit(lambda: (time.sleep(3600), run_maintenance()))
                )
        
        # Start first maintenance task
        if self.multithread_operations and self.thread_pool:
            self.background_tasks.append(self.thread_pool.submit(run_maintenance))
    
    def _compress_old_memories(self, max_memories: int = 100):
        """
        Compress old memories based on age to save storage.
        
        Args:
            max_memories: Maximum number of memories to process in one batch
        """
        if not self.use_lifetime_storage:
            return
        
        # Get archive stats
        stats = self.storage.get_storage_stats()
        
        # Skip if no archived memories
        if stats['archive_memories'] == 0:
            return
        
        # Get list of memories to check for compression
        memories_to_check = []
        
        # Sample from archives, oldest first
        for year in sorted(stats['archive_breakdown'].keys()):
            for month in sorted(stats['archive_breakdown'][year].keys()):
                time_period = f"{year}_{month}"
                period_dir = os.path.join(self.archive_path, time_period)
                
                # Get memory IDs for this period
                memory_files = glob.glob(os.path.join(period_dir, "*.json"))
                memory_ids = [os.path.basename(f).split('.')[0] for f in memory_files]
                
                # Sample memories from this period
                sample_size = min(len(memory_ids), max_memories // 10)
                if sample_size > 0:
                    sampled_ids = np.random.choice(memory_ids, size=sample_size, replace=False)
                    for memory_id in sampled_ids:
                        # Load metadata to check compression level
                        metadata_path = os.path.join(period_dir, f"{memory_id}.json")
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Check if compressed and at what level
                            current_level = metadata.get('storage', {}).get('compression_level', 'none')
                            timestamp = metadata.get('timestamp', 0)
                            
                            # Determine appropriate compression level for age
                            appropriate_level = self.storage.get_compression_level_for_age(timestamp)
                            
                            # If current level is different from appropriate level, add to list
                            if current_level != appropriate_level:
                                memories_to_check.append((memory_id, period_dir, current_level, appropriate_level))
                        except Exception as e:
                            print(f"Error checking compression for memory {memory_id}: {e}")
                
                # If we have enough memories to check, stop sampling
                if len(memories_to_check) >= max_memories:
                    break
            
            # Break outer loop too
            if len(memories_to_check) >= max_memories:
                break
        
        # Process memories that need recompression
        for memory_id, location, current_level, new_level in memories_to_check:
            try:
                if current_level == new_level:
                    continue
                
                # Load memory
                embedding_path = os.path.join(location, f"{memory_id}.pt")
                metadata_path = os.path.join(location, f"{memory_id}.json")
                
                if not os.path.exists(embedding_path) or not os.path.exists(metadata_path):
                    continue
                
                # Check if compressed
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                is_compressed = metadata.get('storage', {}).get('compressed', False)
                
                # Load embedding
                if not is_compressed or current_level == 'none':
                    # Regular torch load
                    embedding = torch.load(embedding_path)
                else:
                    # Load compressed data
                    with open(embedding_path, 'rb') as f:
                        compressed_data = f.read()
                    
                    # Decompress
                    embedding = self.storage.compression_service.decompress(compressed_data, level=current_level)
                
                # Recompress with new level
                if new_level == 'none':
                    # Save without compression
                    torch.save(embedding, embedding_path)
                else:
                    # Compress and save
                    compressed = self.storage.compression_service.compress(embedding, level=new_level)
                    with open(embedding_path, 'wb') as f:
                        f.write(compressed)
                
                # Update metadata
                if 'storage' not in metadata:
                    metadata['storage'] = {}
                
                metadata['storage']['compressed'] = new_level != 'none'
                metadata['storage']['compression_level'] = new_level
                metadata['storage']['recompressed_date'] = time.time()
                
                # Save metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                print(f"Recompressed memory {memory_id} from {current_level} to {new_level}")
            
            except Exception as e:
                print(f"Error recompressing memory {memory_id}: {e}")
    
    def consolidate_memories(self, max_batch_size: int = 100):
        """
        Consolidate similar memories to reduce redundancy.
        This is a form of memory optimization to maximize meaningful storage.
        
        Args:
            max_batch_size: Maximum batch size for processing
        """
        with self.memory_lock:
            # Skip if not enough memories
            if len(self.memories) < 10:
                return
            
            # Get a batch of memories to check
            batch_size = min(len(self.memories), max_batch_size)
            batch_indices = np.random.choice(len(self.memories), size=batch_size, replace=False)
            batch = [self.memories[i] for i in batch_indices]
            
            # Extract embeddings
            batch_embeddings = torch.stack([memory.embedding for memory in batch])
            
            # Calculate similarity matrix
            similarities = torch.matmul(batch_embeddings, batch_embeddings.t())
            
            # Find pairs with high similarity
            similar_pairs = []
            for i in range(batch_size):
                for j in range(i+1, batch_size):
                    similarity = similarities[i, j].item()
                    if similarity > 0.95:  # High similarity threshold
                        similar_pairs.append((i, j, similarity))
            
            # Sort pairs by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Process pairs
            processed_indices = set()
            for i, j, similarity in similar_pairs:
                if i in processed_indices or j in processed_indices:
                    continue
                
                # Get memories
                memory_i = batch[i]
                memory_j = batch[j]
                
                # Determine which to keep (the one with higher surprise or access count)
                importance_i = self._calculate_memory_importance(memory_i)
                importance_j = self._calculate_memory_importance(memory_j)
                
                if importance_i >= importance_j:
                    keep, discard = memory_i, memory_j
                else:
                    keep, discard = memory_j, memory_i
                
                # Update keep memory's metadata to include discard's info
                keep.metadata['consolidated_from'] = keep.metadata.get('consolidated_from', []) + [discard.id]
                keep.metadata['last_consolidated'] = time.time()
                
                # Forget the discarded memory
                self.forget_memory(discard.id)
                
                # Mark as processed
                processed_indices.add(i)
                processed_indices.add(j)
    
    def forward(self, 
               query_embedding: torch.Tensor, 
               top_k: int = 5,
               integration_mode: str = "combined") -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Enhanced forward pass using Titans architecture.
        Retrieves relevant memory embeddings and metadata for a query.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Maximum number of relevant memories to return
            integration_mode: Mode for memory integration
                             "combined": Use both neural and traditional memory
                             "neural_only": Use only neural memory
                             "traditional_only": Use only traditional memory
            
        Returns:
            Tuple of (list of memory embeddings, list of metadata dictionaries)
        """
        use_neural = integration_mode in ["combined", "neural_only"]
        use_traditional = integration_mode in ["combined", "traditional_only"]
        
        # Get relevant memories
        relevant_memories = self.retrieve_relevant_memories(
            query_embedding, 
            top_k=top_k,
            use_neural=use_neural
        )
        
        # Get memory embeddings and metadata
        embeddings = [memory.embedding for memory in relevant_memories]
        metadata = [{
            'surprise_level': memory.surprise_level,
            'timestamp': memory.timestamp,
            'agent_info_id': memory.agent_info_id,
            'metadata': memory.metadata
        } for memory in relevant_memories]
        
        # Also get persistent memory (input-independent)
        if use_neural:
            batch_size = 1
            if query_embedding.dim() > 1 and query_embedding.size(0) > 1:
                batch_size = query_embedding.size(0)
            
            persistent_embeddings = self.persistent_memory(batch_size).squeeze(0)
            
            # Add persistent memory embeddings and metadata
            for i in range(persistent_embeddings.size(0)):
                embeddings.append(persistent_embeddings[i])
                metadata.append({
                    'surprise_level': 0.5,  # Medium surprise level for persistent memory
                    'timestamp': time.time(),
                    'agent_info_id': None,
                    'metadata': {'source': 'persistent_memory', 'index': i}
                })
        
        return embeddings, metadata
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the memory system.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = self.stats.to_dict()
        
        # Add additional storage stats if using lifetime storage
        if self.use_lifetime_storage:
            storage_stats = self.storage.get_storage_stats()
            stats.update(storage_stats)
        
        # Add active memory info
        stats['active_memory_size'] = len(self.memories)
        stats['active_memory_capacity'] = self.max_active_memories
        stats['active_memory_usage_percent'] = (len(self.memories) / self.max_active_memories) * 100
        
        # Add configuration info
        stats['embedding_dim'] = self.embedding_dim
        stats['total_capacity'] = self.capacity
        stats['use_faiss'] = self.use_faiss
        stats['memory_retention_policy'] = self.memory_retention_policy
        
        # Add time-based info
        now = time.time()
        stats['age_in_days'] = (now - self.stats.creation_time) / (24 * 3600)
        stats['days_since_last_access'] = (now - self.stats.last_access_time) / (24 * 3600)
        
        return stats
    
    def clear_memory(self):
        """Clear all memories from the system."""
        with self.memory_lock:
            # Clear active memories
            self.memories = []
            self.memory_embeddings = torch.zeros((0, self.embedding_dim), dtype=torch.float32)
            
            # Reset FAISS index
            if self.use_faiss:
                if isinstance(self.index, faiss.IndexFlatL2):
                    self.index.reset()
                else:
                    self.index = self._create_billion_scale_index(self.embedding_dim)
            
            # Clear cache
            self.cache = LRUCache(min(self.max_active_memories // 10, 10000))
            
            # For lifetime storage, we would need a more complex approach
            # This would typically involve marking files for deletion rather than
            # immediately deleting everything
            
            # Update stats
            self.stats = LifetimeMemoryStats()
    
    def optimize_storage(self):
        """Optimize storage to minimize space usage."""
        if not self.use_lifetime_storage:
            return
        
        with self.memory_lock:
            # 1. Compress old memories
            self._compress_old_memories(max_memories=1000)
            
            # 2. Consolidate similar memories
            self.consolidate_memories(max_batch_size=500)
            
            # 3. Archive old memories if needed
            if len(self.memories) > self.max_active_memories * 0.8:
                self._archive_old_memories()
            
            # 4. Optimize FAISS indexes
            # (This would be implementation-specific)
            
            # 5. Save configuration
            self._save_config()
    
    def save_on_shutdown(self):
        """
        Save all memory data on system shutdown.
        This ensures that all TOVA patterns and compressed memories are properly persisted
        when the program is terminated, so they can be loaded in the next session.
        """
        print("Saving episodic memory state on shutdown...")
        
        # 1. Save neural memory TOVA patterns
        if hasattr(self, 'neural_memory') and self.neural_memory.use_tova:
            self.neural_memory.save_on_shutdown()
        
        # 2. Save configuration
        self._save_config()
        
        # 3. Save archive index
        self._save_archive_index()
        
        # 4. Ensure any pending memory operations are completed
        with self.memory_lock:
            # If there are any memories that haven't been saved yet, save them
            for memory in self.memories:
                if self.use_lifetime_storage and 'storage_location' not in memory.metadata:
                    location = self.storage.store_memory(
                        memory_id=memory.id,
                        embedding=memory.embedding,
                        metadata=memory.to_dict(),
                        timestamp=memory.timestamp
                    )
                    memory.metadata['storage_location'] = location
        
        print("Episodic memory state successfully saved. All memories will be available in the next session.")

'''
episodic_memory.save_on_shutdown() # This needs to be called every time the program is shutting down to be sure that the episodic memory is saved and stored. 
The episodic memory should be stored in the model_save/memory_archive/...file and model_save/episodic_memory_config.json and model_save/memory_archive_/active/...file.

How much memory this episodic memory program can hold and system requirements in detail: 

The episodic memory system is designed for extremely long-term storage, capable of holding decades or even centuries of continuous conversation. Here's a detailed breakdown of its capacity:

Raw Capacity
The default configuration sets a capacity of 10 billion memories (capacity = 10_000_000_000)
Active memory capacity is 10 million memories (max_active_memories = 10_000_000)
Estimating Years of Conversation
If we assume a continuous conversation scenario:

Average conversation: ~15 utterances/memories per minute
That's 900 memories per hour of conversation
For 24/7 continuous conversation: 21,600 memories per day
Annual memory creation: ~7.9 million memories per year
At this rate, the raw capacity of 10 billion memories would last approximately 1,268 years of continuous conversation.

Extended Capacity Through Compression
The system uses a tiered compression strategy that significantly extends this capacity:

Recent memories (<30 days): No compression
30 days to 1 year: Light compression (90% of original size)
1-5 years: Medium compression (50% of original size)
5-20 years: Heavy compression (20% of original size)
20 years: Extreme compression (10% of original size)

With this compression strategy, the effective capacity is much larger. If we assume an average compression ratio of 5:1 across all archived memories (which is conservative), the effective capacity increases to over 6,000 years of continuous conversation.

Memory Management
The system also implements intelligent memory management:

Less important memories are archived first
Similar memories are consolidated to reduce redundancy
Memory importance is calculated based on surprise level, access frequency, and recency
The hierarchical storage system organizes memories by time period for efficient retrieval
Storage Requirements
For storage space requirements:

Each uncompressed memory embedding (768 dimensions) requires about 3KB
10 billion memories would require approximately 30TB of storage in raw form
With compression, this could be reduced to 6-10TB for the entire lifetime capacity
The system is designed to scale to human lifetime and beyond, so it should be more than sufficient for most applications, even with continuous 24/7 operation over many years.

'''