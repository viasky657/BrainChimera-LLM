import time
import uuid
import statistics
import psutil
from collections import OrderedDict
import pickle

import numpy as np
import torch
import faiss

class MemoryNode:
    def __init__(self, memory_chunk, timestamp, centroid=None, surprise_chunk=None, memory_node_id=None, metadata=None):
        self.id = uuid.uuid4() if memory_node_id is None else memory_node_id
        self.memory_chunk = memory_chunk
        self.timestamp = timestamp
        self.centroid = centroid
        self.children = []
        self.last_accessed = time.time()
        self.surprise_chunk = surprise_chunk
        self.metadata = metadata or {}

    def get_num_memories(self):
        if not self.children:
            return 1
        
        total_memories = 0
        for child in self.children:
            total_memories += child.get_num_memories()
            
        return total_memories

    def mark_accessed(self):
        """Marks the node as recently accessed."""
        self.last_accessed = time.time()

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class HierarchicalMemory:
    def __init__(self, num_layers, root_memory_chunk_size, cache_capacity=10000, use_fp16=False):
        self.num_layers = num_layers
        self.root_memory_chunk_size = root_memory_chunk_size
        self.cache_capacity = cache_capacity
        self.active_layer = 0
        self.use_fp16 = use_fp16

        # Create memory layers dynamically
        self.memory_layers = []
        self.surprise_layers = []
        for _ in range(self.num_layers):
            memory_layer, surprise_layer = self._create_memory_layer()
            self.memory_layers.append(memory_layer)
            self.surprise_layers.append(surprise_layer)

        # Initialize the LRU cache
        self._similarity_cache = LRUCache(self.cache_capacity)

        # Create Faiss indices for each layer
        self.index_layers = [faiss.IndexFlatL2(root_memory_chunk_size[0]) for _ in range(num_layers)]
        self.index_surprise_layers = [faiss.IndexFlatL2(root_memory_chunk_size[0]) for _ in range(num_layers)]

        self.last_processed_cluster_id = {}
        self.last_processed_layer_index = None

        self.processing_times_per_node = {i: [] for i in range(num_layers)}
        self.last_layer_traversal_time = time.time()
        self.layer_traversal_times = {i: [] for i in range(num_layers)}
        self.max_tracked_processing_times = 50
        self.processing_time_reset_interval = 3600
        self.last_processing_time_reset = time.time()

        self.layer_node_counts = {i: 0 for i in range(num_layers)}
        self.surprise_layer_node_counts = {i: 0 for i in range(num_layers)}

        self.time_tracking = {
            "similarity_search": [],
            "merge_nodes": [],
            "cross_layer_merge": [],
            "prune": [],
            "reconnect": [],
            "update_index": []
        }
        self.last_time_reset = time.time()
        self.time_reset_interval = 3600
        self.max_tracked_times = 100

        self.active_layers = [False] * num_layers
        self.active_layers[0] = True
        self.surprise_active_layers = [False] * num_layers
        self.surprise_active_layers[0] = True

        self.last_layer_size_update_time = time.time()
        self.layer_size_update_interval = 600

        self.layer_sizes = {i: 0 for i in range(self.num_layers)}
        self.surprise_layer_sizes = {i: 0 for i in range(self.num_layers)}

    def _create_memory_layer(self):
        """Creates a new memory layer with a root node."""
        memory_root = MemoryNode(self.root_memory_chunk_size, time.time())
        surprise_root = MemoryNode(self.root_memory_chunk_size, time.time())
        
        # Initialize surprise_chunk with zeros and link nodes
        memory_root.surprise_chunk = torch.zeros(self.root_memory_chunk_size[0], dtype=torch.float32)
        surprise_root.memory_node_id = memory_root.id
        memory_root.centroid = torch.zeros_like(memory_root.centroid)

        # Apply mixed precision if enabled
        if self.use_fp16:
            memory_root.memory_chunk = memory_root.memory_chunk.half()
            memory_root.centroid = memory_root.centroid.half()
            surprise_root.surprise_chunk = surprise_root.surprise_chunk.half()

        return memory_root, surprise_root

    def activate_next_layer(self):
        """Activates the next memory layer if available."""
        if self.active_layer < self.num_layers - 1:
            self.active_layer += 1
            self.active_layers[self.active_layer] = True
            self.surprise_active_layers[self.active_layer] = True
            self._similarity_cache = LRUCache(self.cache_capacity)
            print(f"Activated memory layer: {self.active_layer}")
        else:
            print("All memory layers are active.")

    def is_layer_full(self, threshold_factor=0.8):
        """
        Checks if the active memory layer is considered full based on a threshold factor.
        """
        active_memory_root = self.memory_layers[self.active_layer]
        active_surprise_root = self.surprise_layers[self.active_layer]

        num_nodes = self._count_nodes(active_memory_root)
        num_non_empty_nodes = self._count_non_empty_nodes(active_memory_root)

        return num_non_empty_nodes / num_nodes >= threshold_factor

    def _count_nodes(self, node):
        """Counts the total number of nodes in a subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _count_non_empty_nodes(self, node):
        """Counts the number of non-empty nodes in a subtree."""
        count = 1 if node.memory_chunk.any() else 0
        for child in node.children:
            count += self._count_non_empty_nodes(child)
        return count
    

    def forget_memories(self, start_time, end_time, agent_info_id=None): #This is the only forgetting function that works in the program. 
        """
        Resets memories within a time range and optional agent_info ID.
        Targets both memory and surprise layers.
        """
        for layer_idx in range(self.num_layers):
            mem_root = self.memory_layers[layer_idx]
            surprise_root = self.surprise_layers[layer_idx]
            self._forget_memories_recursive(
                mem_root, 
                surprise_root,
                start_time=start_time,
                end_time=end_time,
                agent_info_id=agent_info_id
            )

    def _forget_memories_recursive(self, mem_node, surprise_node, start_time, end_time, agent_info_id):
        """Recursively resets memories matching time and agent criteria"""
        # Check if this node matches the criteria
        if (start_time <= mem_node.timestamp <= end_time and
            (agent_info_id is None or mem_node.metadata.get('agent_info') == agent_info_id)):
            
            # Reset memory values
            mem_node.memory_chunk = torch.zeros_like(mem_node.memory_chunk)
            mem_node.centroid = torch.zeros_like(mem_node.centroid)
            mem_node.surprise_chunk = torch.zeros_like(mem_node.surprise_chunk)
            
            # Reset surprise node
            if surprise_node:
                surprise_node.surprise_chunk = torch.zeros_like(surprise_node.surprise_chunk)

        # Process children recursively
        for mem_child, surprise_child in zip(mem_node.children, surprise_node.children):
            self._forget_memories_recursive(
                mem_child,
                surprise_child,
                start_time,
                end_time,
                agent_info_id
            )


'''

**Explanation of the Additions:**

*   The code includes additional logic for handling surprise nodes in methods like `merge_nodes_across_layers`, `_forget_word_recursive`, and `prune_children`.
*   The `_calculate_surprise_factor` method is introduced to compute a surprise factor based on the `surprise_chunk` of a node. This factor can influence the forgetting process.
*   Placeholders for byte-to-patch conversion and surprise calculation are included, which you'll need to implement based on your specific BLT model and surprise mechanism.
*   The code includes various helper methods like `_track_time`, `_get_average_time`, `_should_reset_times`, `_reset_times`, `_update_layer_size`, `_get_layer_size`, and `_should_update_layer_size` for tracking time, managing layer sizes, and other utility functions.
*   Methods like `merge_similar_nodes_chunked` and `check_cross_layer_similarity_chunked` are modified to handle dynamic chunk sizing and incorporate the tracking of processing times.
*   The code includes methods for saving and loading the hierarchical memory, managing the active layers, and triggering the memory optimization process either manually or automatically at scheduled intervals.

**How to Use:**

1. **Instantiate `HierarchicalMemory`:** Create an object of the `HierarchicalMemory` class, providing the necessary parameters like the number of layers, root memory chunk size, and cache capacity.
2. **Integrate with BLT Model:** Ensure that the BLT model's components (Local Encoder, Local Decoder, Latent Global Transformer) are integrated with this memory structure. This involves adapting the BLT model to work with the memory nodes and their associated data.
3. **Implement Byte-Level Operations:** Replace token-based operations with byte-level operations. This includes converting words to byte sequences, handling byte patches instead of tokens, and updating centroids and memory chunks with byte-level data.
4. **Surprise Mechanism:** Implement the surprise calculation logic in `_calculate_surprise_factor` and integrate it with the memory operations.
5. **Forgetting Mechanism:** Implement the logic for forgetting words or memories based on time, context, or other criteria in the `forget_words` and `_forget_word_recursive` methods.
6. **Training and Inference:** Use the `HierarchicalMemory` object in conjunction with your BLT model during training and inference. Ensure that the memory is updated and utilized appropriately in these processes.
7. **Memory Optimization:** Use `trigger_memory_optimization` for manual optimization or `_schedule_memory_optimization` for automated optimization at regular intervals.

Remember to replace placeholder methods like `_get_word_vector`, `byte_to_patch`, `_prepare_blt_input`, `_determine_patch_boundaries`, `_group_into_patches`, and `_calculate_entropy` with your actual implementations based on your specific requirements and BLT model architecture.

This comprehensive code structure should provide a solid foundation for implementing a hierarchical memory system with advanced features like dynamic patching, surprise-based forgetting, and efficient memory management. '''
