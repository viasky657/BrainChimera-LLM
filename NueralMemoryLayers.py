
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
    def __init__(self, memory_chunk, timestamp, centroid=None, surprise_chunk=None, memory_node_id=None):
        self.id = uuid.uuid4() if memory_node_id is None else memory_node_id
        self.memory_chunk = memory_chunk
        self.timestamp = timestamp
        self.centroid = centroid
        self.children = []
        self.last_accessed = time.time()
        self.surprise_chunk = surprise_chunk

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

    def save(self, filepath):
        """Saves the hierarchical memory to a file."""
        # Don't save the cache
        cache_holder = self._similarity_cache
        self._similarity_cache = None

        # Save the Faiss indices separately
        for i, index in enumerate(self.index_layers):
            faiss.write_index(index, f"{filepath}.layer_{i}.index")
        for i, index in enumerate(self.index_surprise_layers):
            faiss.write_index(index, f"{filepath}.surprise_layer_{i}.index")

        # Remove the indices before pickling
        index_layers_holder = self.index_layers
        index_surprise_layers_holder = self.index_surprise_layers
        self.index_layers = None
        self.index_surprise_layers = None

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        # Restore the cache and indices
        self._similarity_cache = cache_holder
        self.index_layers = index_layers_holder
        self.index_surprise_layers = index_surprise_layers_holder

    @staticmethod
    def load(filepath, cache_capacity=10000):
        """Loads a hierarchical memory from a file."""
        with open(filepath, 'rb') as f:
            loaded_memory = pickle.load(f)

        # Generate new unique IDs for all nodes in all layers
        def regenerate_ids(node):
            node.id = uuid.uuid4()
            for child in node.children:
                regenerate_ids(child)

        for layer in loaded_memory.memory_layers:
            regenerate_ids(layer)
        for layer in loaded_memory.surprise_layers:
            regenerate_ids(layer)

        # Initialize a new LRU cache for the active layer
        loaded_memory._similarity_cache = LRUCache(cache_capacity)

        # Load the Faiss indices
        loaded_memory.index_layers = []
        loaded_memory.index_surprise_layers = []
        for i in range(loaded_memory.num_layers):
            index = faiss.read_index(f"{filepath}.layer_{i}.index")
            loaded_memory.index_layers.append(index)

            surprise_index = faiss.read_index(f"{filepath}.surprise_layer_{i}.index")
            loaded_memory.index_surprise_layers.append(surprise_index)

        return loaded_memory

    def merge_similar_nodes(self, similarity_threshold):
        """
        Recursively merges similar nodes in the active memory layer, including the surprise hierarchy.
        Uses an LRU cache with unique node IDs for efficiency.
        """
        active_memory_root = self.memory_layers[self.active_layer]
        active_surprise_root = self.surprise_layers[self.active_layer]

        self._similarity_cache = LRUCache(self.cache_capacity)
        self._merge_similar_nodes_recursive(active_memory_root, active_surprise_root, similarity_threshold)
        self._similarity_cache = None

    def _merge_similar_nodes_recursive(self, node, surprise_node, similarity_threshold):
        if not node.children:
            return

        # Recursively merge similar nodes in the children
        for i in range(len(node.children)):
            self._merge_similar_nodes_recursive(node.children[i], surprise_node.children[i], similarity_threshold)

        # Compare children of the current node (with caching)
        merged = [False] * len(node.children)
        for i in range(len(node.children)):
            if merged[i]:
                continue
            for j in range(i + 1, len(node.children)):
                if merged[j]:
                    continue

                # Track time for similarity search
                similarity_search_start_time = time.time()
                node_pair = tuple(sorted((node.children[i].id, node.children[j].id)))
                similarity = self._similarity_cache.get(node_pair)
                if similarity is None:
                    similarity = torch.dot(node.children[i].centroid, node.children[j].centroid)
                    self._similarity_cache.put(node_pair, similarity)
                similarity_search_elapsed_time = time.time() - similarity_search_start_time
                self._track_time("similarity_search", similarity_search_elapsed_time)

                if similarity >= similarity_threshold:
                    # Merge memory nodes
                    merge_start_time = time.time()
                    new_node = self.merge_nodes(node.children[i], node.children[j])
                    merge_elapsed_time = time.time() - merge_start_time
                    self._track_time("merge_nodes", merge_elapsed_time)

                    # Merge corresponding surprise nodes
                    new_surprise_node = self.merge_surprise_nodes(surprise_node.children[i], surprise_node.children[j])

                    # Replace the merged nodes in the children lists
                    new_children = [child for k, child in enumerate(node.children) if k != i and k != j]
                    new_children.append(new_node)
                    node.children = new_children

                    new_surprise_children = [child for k, child in enumerate(surprise_node.children) if k != i and k != j]
                    new_surprise_children.append(new_surprise_node)
                    surprise_node.children = new_surprise_children

                    merged[i] = True
                    merged[j] = True
                    break

    def merge_nodes(self, node1, node2):
        """
        Merges two memory nodes into a single new node, weighting the average by the inverse
        of the timestamps to give more weight to recent memories.
        """
        # Calculate weights based on the inverse of timestamps
        epsilon = 1e-10
        weight1 = 1 / (node1.timestamp + epsilon)
        weight2 = 1 / (node2.timestamp + epsilon)
        total_weight = weight1 + weight2
        weight1 /= total_weight
        weight2 /= total_weight

        # Create a new memory chunk by weighted averaging
        new_memory_chunk = (node1.memory_chunk * weight1 + node2.memory_chunk * weight2)

        # Set the timestamp of the new node using weighted averaging
        new_timestamp = (node1.timestamp * weight1 + node2.timestamp * weight2)

        # Set the centroid of the new node using weighted averaging
        new_centroid = (node1.centroid * weight1 + node2.centroid * weight2)

        # Create the new merged node
        new_node = MemoryNode(new_memory_chunk, new_timestamp, new_centroid)
        new_node.memory_node_id = node2.id

        # Make the children of the two merged nodes the children of the new node
        new_node.children = node1.children + node2.children

        # Update Faiss index for the new node
        layer_index = self._find_layer_index(node1)
        if layer_index is not None:
            self._add_node_to_index(new_node, layer_index)

        return new_node

    def merge_surprise_nodes(self, node1, node2):
        """
        Merges two surprise nodes into a single new node.
        """
        # Merge surprise_chunks, weighting by recency or other criteria
        epsilon = 1e-10
        weight1 = 1 / (node1.timestamp + epsilon)
        weight2 = 1 / (node2.timestamp + epsilon)
        total_weight = weight1 + weight2
        new_surprise_chunk = (node1.surprise_chunk * weight1 + node2.surprise_chunk * weight2) / total_weight

        new_node = MemoryNode(
            memory_chunk=None,
            timestamp=None,
            centroid=None,
            surprise_chunk=new_surprise_chunk
        )
        new_node.children = node1.children + node2.children
        return new_node

    def check_cross_layer_similarity(self, similarity_threshold, time_threshold, decay_factor=0.5):
        """
        Checks for similar memories across layers and decays or removes them based on
        similarity and time thresholds.
        """
        for i in range(self.num_layers):
            # Skip inactive layers
            if not self.active_layers[i]:
                print(f"Skipping inactive layer {i}")
                continue
            for j in range(i + 1, self.num_layers):
                # Skip inactive layers
                if not self.active_layers[j]:
                    print(f"Skipping inactive layer {j}")
                    continue

                print(f"Checking similarity between layers {i} and {j}")
                self._compare_layers(
                    self.memory_layers[i],
                    self.surprise_layers[i],
                    self.memory_layers[j],
                    self.surprise_layers[j],
                    similarity_threshold,
                    time_threshold,
                    decay_factor
                )

    def _compare_layers(self, layer1_root, surprise_layer1_root, layer2_root, surprise_layer2_root, similarity_threshold, time_threshold, decay_factor):
        """
        Compares nodes between two layers and decays or removes similar memories based on thresholds.
        """
        for node1 in self._traverse_layer(layer1_root):
            for node2 in self._traverse_layer(layer2_root):
                similarity = torch.dot(node1.centroid, node2.centroid)
                time_diff = abs(node1.timestamp - node2.timestamp)

                if similarity >= similarity_threshold and time_diff <= time_threshold:
                    print(f"  Found similar nodes across layers (similarity: {similarity:.2f}, time diff: {time_diff:.2f})")

                    layer1_index = self.memory_layers.index(layer1_root)
                    layer2_index = self.memory_layers.index(layer2_root)

                    # Merge nodes across layers
                    cross_layer_merge_start_time = time.time()
                    merged_node = self.merge_nodes_across_layers(node1, layer1_index, node2, layer2_index, similarity_threshold, surprise_layer1_root, surprise_layer2_root)
                    cross_layer_merge_elapsed_time = time.time() - cross_layer_merge_start_time
                    self._track_time("cross_layer_merge", cross_layer_merge_elapsed_time)

    def _traverse_layer(self, node):
        """
        Traverses all nodes in a layer using depth-first search.
        """
        stack = [node]
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(node.children):
                stack.append(child)
    
    def merge_nodes_across_layers(self, node1, layer1_index, node2, layer2_index, similarity_threshold, surprise_layer1_root, surprise_layer2_root):
        """
        Merges two similar nodes from different layers into a single node in the more recent layer.
        """
        # Ensure node2 is in the more recent layer
        if layer1_index > layer2_index:
            node1, node2 = node2, node1
            layer1_index, layer2_index = layer2_index, layer1_index
            surprise_layer1_root, surprise_layer2_root = surprise_layer2_root, surprise_layer1_root

        # Calculate weights (favor more recent node)
        epsilon = 1e-10
        weight1 = 1 / (node1.timestamp + epsilon)
        weight2 = 1 / (node2.timestamp + epsilon)
        total_weight = weight1 + weight2
        weight1 /= total_weight
        weight2 /= total_weight

        # Merge node data into node2 (more recent layer)
        new_memory_chunk = (node1.memory_chunk * weight1 + node2.memory_chunk * weight2)
        new_timestamp = (node1.timestamp * weight1 + node2.timestamp * weight2)
        new_centroid = (node1.centroid * weight1 + node2.centroid * weight2)

        # Create new node in more recent layer
        merged_node = MemoryNode(new_memory_chunk, new_timestamp, new_centroid)
        merged_node.children = node2.children  # Start with children from more recent node

        # Find corresponding surprise nodes and merge them
        surprise_node1 = self._find_corresponding_surprise_node(node1, surprise_layer1_root)
        surprise_node2 = self._find_corresponding_surprise_node(node2, surprise_layer2_root)

        if surprise_node1 is not None and surprise_node2 is not None:
            merged_surprise_node = self.merge_surprise_nodes(surprise_node1, surprise_node2)
            merged_surprise_node.memory_node_id = merged_node.id  # Link the merged surprise node to the merged memory node
            merged_node.surprise_chunk = merged_surprise_node  # Link the merged memory node to the merged surprise node
            # Replace the old surprise nodes with the new merged one in the more recent layer
            self._replace_node(surprise_node2, merged_surprise_node, self.surprise_layers[layer2_index])

        # Handle children (recursively merge similar children)
        for child1 in node1.children:
            most_similar_child2 = None
            highest_similarity = -1

            for child2 in node2.children:
                similarity_search_start_time = time.time()
                similarity = torch.dot(child1.centroid, child2.centroid)
                similarity_search_elapsed_time = time.time() - similarity_search_start_time
                self._track_time("similarity_search", similarity_search_elapsed_time)
                if similarity >= similarity_threshold and similarity > highest_similarity:
                    most_similar_child2 = child2
                    highest_similarity = similarity

            if most_similar_child2 is not None:
                # Find corresponding surprise nodes for children
                surprise_child1 = self._find_corresponding_surprise_node(child1, surprise_layer1_root)
                surprise_child2 = self._find_corresponding_surprise_node(most_similar_child2, surprise_layer2_root)
                
                # Recursively merge child1 with the most similar child2
                recursive_merge_start_time = time.time()
                merged_child = self.merge_nodes_across_layers(child1, layer1_index, most_similar_child2, layer2_index, similarity_threshold, surprise_child1, surprise_child2)
                recursive_merge_elapsed_time = time.time() - recursive_merge_start_time
                self._track_time("merge_nodes", recursive_merge_elapsed_time)

                # Remove the merged child from node2's children (it's now part of merged_node)
                node2.children.remove(most_similar_child2)

                # Also remove the corresponding surprise child from surprise_node2's children
                if surprise_child2 is not None:
                    surprise_node2.children.remove(surprise_child2)
            else:
                # Attach child1 to merged_node (no similar child found in node2)
                merged_node.children.append(child1)

        # Update parent-child relationships (remove node1 from its parent)
        self._remove_node_from_parent(node1, self.memory_layers[layer1_index])

        # Add merged_node to the more recent layer (if not already present)
        if merged_node not in self.memory_layers[layer2_index].children:
            self.memory_layers[layer2_index].children.append(merged_node)

        # Prune children of the merged node (optional)
        prune_start_time = time.time()
        self.prune_children(merged_node, layer2_index, 0.8, 0.7)
        prune_elapsed_time = time.time() - prune_start_time
        self._track_time("prune", prune_elapsed_time)

        # Update Faiss index for the merged node
        update_index_start_time = time.time()
        self._update_node_in_index(merged_node, layer2_index)
        update_index_elapsed_time = time.time() - update_index_start_time
        self._track_time("update_index", update_index_elapsed_time)

        # Update layer sizes after merging
        self._update_layer_node_count(layer1_index, -1)  # Removed one node from layer1
        self._update_layer_node_count(layer2_index, 1)

        return merged_node

    def _find_corresponding_surprise_node(self, memory_node, surprise_layer_root):
        """
        Finds the corresponding surprise node for a given memory node.
        """
        memory_node_id = memory_node.id
        for surprise_node in self._traverse_layer(surprise_layer_root):
            if hasattr(surprise_node, 'memory_node_id') and surprise_node.memory_node_id == memory_node_id:
                return surprise_node
        return None

    def _replace_node(self, old_node, new_node, layer_root):
        """
        Replaces an old node with a new node in the hierarchy.
        """
        for parent in self._traverse_layer(layer_root):
            for i, child in enumerate(parent.children):
                if child.id == old_node.id:
                    parent.children[i] = new_node
                    return
    
    def _remove_node_from_parent(self, node, layer_root):
        """
        Removes a node from its parent's children list.
        """
        for parent in self._traverse_layer(layer_root):
            if node in parent.children:
                parent.children.remove(node)
                return

    def _find_node_by_id(self, node_id, layer_root):
        """
        Finds a node in a layer by its unique ID.
        """
        for node in self._traverse_layer(layer_root):
            if node.id == node_id:
                return node
        return None

    def reconnect_node(self, node, layer_index, similarity_threshold):
        """
        Reconnects a node within the hierarchy based on similarity to existing clusters.
        """
        layer_root = self.memory_layers[layer_index]

        # Track time for similarity search
        similarity_search_start_time = time.time()
        query_vector = np.array([node.centroid.detach().cpu().numpy()], dtype=np.float32)
        k = 1
        distances, indices = self.index_layers[layer_index].search(query_vector, k)
        similarity_search_elapsed_time = time.time() - similarity_search_start_time
        self._track_time("similarity_search", similarity_search_elapsed_time)

        # Find the best match (excluding itself)
        best_match_index = None
        highest_similarity = -1
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                candidate_node = self._get_node_by_index(idx, layer_root)
                if candidate_node.id != node.id:
                    similarity = torch.dot(node.centroid, candidate_node.centroid)
                    if similarity >= similarity_threshold and similarity > highest_similarity:
                        best_match_index = idx
                        highest_similarity = similarity

        # Reconnect to the best match if found
        if best_match_index is not None:
            best_match = self._get_node_by_index(best_match_index, layer_root)
            # Remove node from its current parent
            self._remove_node_from_parent(node, layer_root)

            # Attach node to the best match
            reconnect_start_time = time.time()
            best_match.children.append(node)
            reconnect_elapsed_time = time.time() - reconnect_start_time
            self._track_time("reconnect", reconnect_elapsed_time)
            print(f"Reconnected node {node.id} to node {best_match.id} (similarity: {highest_similarity:.2f})")
        else:
            print(f"Could not find a suitable match for node {node.id} within the similarity threshold.")

    def _get_node_by_index(self, index, layer_root):
        """
        Retrieves a node from a layer based on its index in the Faiss index.
        """
        for i, node in enumerate(self._traverse_layer(layer_root)):
            if i == index:
                return node
        return None

    def _get_average_processing_time(self, category):
        """
        Gets the average processing time for a specific operation category.
        """
        if category not in self.time_tracking or not self.time_tracking[category]:
            return 0.1  # Default value if no data available
        return statistics.mean(self.time_tracking[category])

    def _calculate_dynamic_chunk_size(self, available_time_seconds, layer_size, processing_time_per_node_seconds, max_chunk_size=500):
        """
        Calculates a dynamic chunk size based on available time, layer size, and processing time per node.
        """
        if self._should_reset_times():
            self._reset_times()

        # Estimate the maximum number of nodes that can be processed within the available time
        merge_time_per_node = self._get_average_processing_time("merge_nodes")
        if merge_time_per_node > 0:
            max_processable_nodes = int(available_time_seconds / merge_time_per_node)
        else:
            max_processable_nodes = int(available_time_seconds / 0.1)  # Default value

        # Calculate a chunk size based on a fraction of the layer size
        chunk_size = min(max(int(layer_size * 0.1), 1), max_processable_nodes, max_chunk_size)

        # Adjust chunk size based on system load (optional)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if cpu_usage > 80:
            chunk_size = int(chunk_size * 0.5)
        elif cpu_usage < 20:
            chunk_size = int(chunk_size * 1.5)

        return max(chunk_size, 1)

    def _track_time(self, category, elapsed_time):
        """Tracks the elapsed time for a specific category."""
        if category not in self.time_tracking:
            self.time_tracking[category] = []

        self.time_tracking[category].append(elapsed_time)

        # Keep only the last 'max_tracked_times' entries
        if len(self.time_tracking[category]) > self.max_tracked_times:
            self.time_tracking[category].pop(0)

    def _get_average_time(self, category):
        """Gets the average time for a specific category."""
        if category not in self.time_tracking or not self.time_tracking[category]:
            return 0.0  # Default value if no data is available

        return statistics.mean(self.time_tracking[category])

    def _should_reset_times(self):
        """Checks if it's time to reset the tracked times."""
        return time.time() - self.last_time_reset > self.time_reset_interval

    def _reset_times(self):
        """Resets the tracked times."""
        self.time_tracking = {category: [] for category in self.time_tracking}
        self.last_time_reset = time.time()

    def _update_layer_size(self, layer_index):
        """Updates the estimated size of a layer."""
        self.layer_sizes[layer_index] = self._count_nodes(self.memory_layers[layer_index])
        self.surprise_layer_sizes[layer_index] = self._count_nodes(self.surprise_layers[layer_index])

    def _get_layer_size(self, layer_index):
        """Gets the estimated size of a layer."""
        return self.layer_sizes.get(layer_index, 0)

    def _should_update_layer_size(self):
        """Checks if it's time to update the layer sizes."""
        return time.time() - self.last_layer_size_update_time > self.layer_size_update_interval

    def _update_processing_time(self, layer_index, elapsed_time):
        """Updates the moving average of processing time per node for a specific layer."""
        if layer_index not in self.processing_times_per_node:
            self.processing_times_per_node[layer_index] = []

        self.processing_times_per_node[layer_index].append(elapsed_time)

        # Keep only the last 'max_tracked_processing_times' entries
        if len(self.processing_times_per_node[layer_index]) > self.max_tracked_processing_times:
            self.processing_times_per_node[layer_index].pop(0)

    def _should_reset_processing_time(self):
        """Checks if it's time to reset the average processing time."""
        return time.time() - self.last_processing_time_reset > self.processing_time_reset_interval

    def _reset_processing_time(self):
        """Resets the average processing time."""
        self.processing_times_per_node = {i: [] for i in range(self.num_layers)}
        self.last_processing_time_reset = time.time()

    def _update_layer_node_count(self, layer_index, delta):
        """Updates the node count for a given layer."""
        self.layer_node_counts[layer_index] += delta

    def _get_layer_node_count(self, layer_index):
        """Gets the current node count for a given layer."""
        return self.layer_node_counts.get(layer_index, 0)  # Return 0 if layer not found

    def merge_similar_nodes_chunked(self, similarity_threshold, available_time_seconds, max_chunk_size=500):
        """
        Merges similar nodes within the active layer in chunks, dynamically adjusting chunk size.
        """
        active_memory_root = self.memory_layers[self.active_layer]
        active_surprise_root = self.surprise_layers[self.active_layer]

        # Load the last processed cluster ID for the active layer
        start_node_id = self.last_processed_cluster_id.get(self.active_layer) if self.last_processed_cluster_id else None

        # Find the starting node based on the ID
        if start_node_id:
            start_node = self._find_node_by_id(start_node_id, active_memory_root)
        else:
            start_node = active_memory_root

        # If the start node is not found (e.g., it was merged earlier), start from the root
        if not start_node:
            start_node = active_memory_root

        # Update layer size if needed
        if self._should_update_layer_size():
            self._update_layer_size(self.active_layer)
            self.last_layer_size_update_time = time.time()

        # Estimate layer size and processing time per node
        layer_size = self._get_layer_size(self.active_layer)
        processing_time_per_node_seconds = self._get_average_processing_time("merge_nodes")

        # Calculate dynamic chunk size
        chunk_size = self._calculate_dynamic_chunk_size(available_time_seconds, layer_size, processing_time_per_node_seconds, max_chunk_size)

        processed_count = 0
        for node in self._traverse_layer_from_node(start_node):
            if processed_count >= chunk_size:
                break

            # Track time for merging
            merge_start_time = time.time()
            self._merge_similar_nodes_recursive(node, active_surprise_root, similarity_threshold)
            merge_elapsed_time = time.time() - merge_start_time
            self._track_time("merge_nodes", merge_elapsed_time)

            # Update the last processed cluster ID for the active layer
            self.last_processed_cluster_id[self.active_layer] = node.id
            processed_count += 1

        # If we've reached the end of the layer, reset the last processed ID
        if processed_count < chunk_size:
            self.last_processed_cluster_id[self.active_layer] = None

        # Update processing time after each node
        self._update_processing_time(self.active_layer, merge_elapsed_time)

    def check_cross_layer_similarity_chunked(self, similarity_threshold, time_threshold, decay_factor=0.5, max_chunk_size=500, available_time_seconds=None):
        """
        Checks for similar memories across layers and merges them in chunks, dynamically adjusting chunk size.
        Skips inactive layers.
        """
        if available_time_seconds is None:
            available_time_seconds = float('inf')

        start_time = time.time()

        start_layer = self.last_processed_layer_index if self.last_processed_layer_index is not None else 0
        for i in range(start_layer, self.num_layers):
            # Skip inactive layers
            if not self.active_layers[i]:
                print(f"Skipping inactive layer {i}")
                continue

            for j in range(i + 1, self.num_layers):
                # Skip inactive layers
                if not self.active_layers[j]:
                    print(f"Skipping inactive layer {j}")
                    continue

                print(f"Checking similarity between layers {i} and {j}")

                # Load the last processed cluster ID for this pair of layers
                last_processed_id = self.last_processed_cluster_id.get((i, j))

                # Update layer sizes if needed
                if self._should_update_layer_size():
                    for k in range(self.num_layers):
                        self._update_layer_size(k)
                    self.last_layer_size_update_time = time.time()

                # Calculate remaining time
                elapsed_time = time.time() - start_time
                remaining_time = available_time_seconds - elapsed_time

                # Estimate layer size and processing time per node
                layer1_size = self._get_layer_size(i)
                layer2_size = self._get_layer_size(j)
                processing_time_per_node_seconds = self._get_average_processing_time("cross_layer_merge")

                # Calculate dynamic chunk size based on the smaller layer size
                chunk_size = self._calculate_dynamic_chunk_size(remaining_time, min(layer1_size, layer2_size), processing_time_per_node_seconds, max_chunk_size)

                if not self._compare_layers_chunked(
                    self.memory_layers[i],
                    self.surprise_layers[i],
                    self.memory_layers[j],
                    self.surprise_layers[j],
                    similarity_threshold,
                    time_threshold,
                    decay_factor,
                    chunk_size,
                    last_processed_id,
                    remaining_time
                ):
                    # Save the progress and return if the chunk is not finished or time is up
                    self.last_processed_layer_index = i
                    return

            # Reset the last processed layer index after finishing a layer
            self.last_processed_layer_index = None

    def _compare_layers_chunked(self, layer1_root, surprise_layer1_root, layer2_root, surprise_layer2_root, similarity_threshold, time_threshold, decay_factor, chunk_size, start_node_id=None, available_time_seconds=None):
        """
        Compares nodes between two layers and merges similar memories based on thresholds in chunks.

        Returns:
            True if the end of a layer is reached, False otherwise.
        """
        # Find the starting node based on the ID
        if start_node_id:
            start_node = self._find_node_by_id(start_node_id, layer1_root)
        else:
            start_node = layer1_root

        # If the start node is not found (e.g., it was merged earlier), start from the root
        if not start_node:
            start_node = layer1_root

        # Traverse the first layer and process nodes in chunks
        processed_count = 0
        start_time = time.time()
        for node1 in self._traverse_layer_from_node(start_node):
            for node2 in self._traverse_layer(layer2_root):
                elapsed_time = time.time() - start_time
                if available_time_seconds is not None and elapsed_time >= available_time_seconds:
                    print("Time limit reached during cross-layer comparison.")
                    layer_pair = (self.memory_layers.index(layer1_root), self.memory_layers.index(layer2_root))
                    self.last_processed_cluster_id[layer_pair] = node1.id
                    return False

                if processed_count >= chunk_size:
                    # Save the last processed cluster ID for this pair of layers
                    layer_pair = (self.memory_layers.index(layer1_root), self.memory_layers.index(layer2_root))
                    self.last_processed_cluster_id[layer_pair] = node1.id
                    return False  # Indicate that the chunk is not finished

                similarity_search_start_time = time.time()
                similarity = torch.dot(node1.centroid, node2.centroid)
                similarity_search_elapsed_time = time.time() - similarity_search_start_time
                self._track_time("similarity_search", similarity_search_elapsed_time)

                time_diff = abs(node1.timestamp - node2.timestamp)

                if similarity >= similarity_threshold and time_diff <= time_threshold:
                    print(f"  Found similar nodes across layers (similarity: {similarity:.2f}, time diff: {time_diff:.2f})")

                    layer1_index = self.memory_layers.index(layer1_root)
                    layer2_index = self.memory_layers.index(layer2_root)

                    # Merge nodes across layers
                    cross_layer_merge_start_time = time.time()
                    merged_node = self.merge_nodes_across_layers(node1, layer1_index, node2, layer2_index, similarity_threshold, surprise_layer1_root, surprise_layer2_root)
                    cross_layer_merge_elapsed_time = time.time() - cross_layer_merge_start_time
                    self._track_time("cross_layer_merge", cross_layer_merge_elapsed_time)

            processed_count += 1

        # If we've reached the end of the layer, reset the last processed ID for this pair
        layer_pair = (self.memory_layers.index(layer1_root), self.memory_layers.index(layer2_root))
        self.last_processed_cluster_id[layer_pair] = None
        return True

    def prune_children(self, node, layer_index, threshold, reconnection_similarity_threshold):
        """
        Prunes children of a node by resetting their memory content and reconnecting them.
        """
        children_to_reset = []
        for child in node.children:
            similarity = torch.dot(node.centroid, child.centroid)
            if similarity < threshold:
                children_to_reset.append(child)

        for child in children_to_reset:
            # Reset the child's memory content
            child.memory_chunk = torch.zeros_like(child.memory_chunk)
            child.timestamp = time.time()  # Update the timestamp
            child.centroid = torch.zeros_like(child.centroid) # Reset centroid

            # Also reset the surprise information
            if hasattr(child, 'surprise_chunk') and child.surprise_chunk is not None:
                child.surprise_chunk = torch.zeros_like(child.surprise_chunk)

            # Reconnect the child to a more relevant cluster
            layer_root = self.memory_layers[layer_index]
            reconnect_start_time = time.time()
            self.reconnect_node(child, layer_index, reconnection_similarity_threshold)
            reconnect_elapsed_time = time.time() - reconnect_start_time
            self._track_time("reconnect", reconnect_elapsed_time)

            # Update the Faiss index
            update_index_start_time = time.time()
            self._update_node_in_index(child, layer_index)
            update_index_elapsed_time = time.time() - update_index_start_time
            self._track_time("update_index", update_index_elapsed_time)

            # Recursively prune the child's subtree (if needed)
            self.prune_children(child, layer_index, threshold, reconnection_similarity_threshold)

    def _add_node_to_index(self, node, layer_index):
        """Adds a node's centroid to the Faiss index of the corresponding layer."""
        # Add to memory index
        add_index_start_time = time.time()
        self.index_layers[layer_index].add(np.array([node.centroid.detach().cpu().numpy()], dtype=np.float32))
        add_index_elapsed_time = time.time() - add_index_start_time
        self._track_time("update_index", add_index_elapsed_time)
        self._update_layer_node_count(layer_index, 1)

    def _update_node_in_index(self, node, layer_index):
        """Updates a node's centroid in the Faiss index of the corresponding layer."""
        # Remove the old vector (if it exists) and add the updated one
        update_index_start_time = time.time()
        self._remove_node_from_index(node, layer_index)
        self._add_node_to_index(node, layer_index)
        update_index_elapsed_time = time.time() - update_index_start_time
        self._track_time("update_index", update_index_elapsed_time)

    def _remove_node_from_index(self, node, layer_index):
        """Removes a node's centroid from the Faiss index of the corresponding layer."""
        remove_index_start_time = time.time()
        new_index = faiss.IndexFlatL2(self.index_layers[layer_index].d)
        for other_node in self._traverse_layer(self.memory_layers[layer_index]):
            if other_node.id != node.id:
                new_index.add(np.array([other_node.centroid.detach().cpu().numpy()], dtype=np.float32))
        self.index_layers[layer_index] = new_index
        remove_index_elapsed_time = time.time() - remove_index_start_time
        self._track_time("update_index", remove_index_elapsed_time)
        self._update_layer_node_count(layer_index, -1)

    def forget_words(self, words_to_forget, similarity_threshold, surprise_threshold=0.5):
        """
        Forgets memories associated with specific words.
        """
        for layer_index in range(self.num_layers):
            layer_root = self.memory_layers[layer_index]
            surprise_layer_root = self.surprise_layers[layer_index]
            for word in words_to_forget:
                word_vector = self._get_word_vector(word)
                self._forget_word_recursive(layer_root, surprise_layer_root, word_vector, similarity_threshold, surprise_threshold)

    def _forget_word_recursive(self, node, surprise_node, word_vector, similarity_threshold, surprise_threshold):
        """
        Recursively traverses the memory hierarchy and forgets nodes associated with a specific word.
        Considers surprise as a modulating factor.
        """
        similarity = torch.dot(node.centroid, word_vector)

        if similarity >= similarity_threshold:
            # Consider surprise when deciding whether to forget
            if hasattr(node, 'surprise_chunk') and node.surprise_chunk is not None:
                surprise_factor = self._calculate_surprise_factor(node.surprise_chunk)
            else:
                surprise_factor = 1.0  # Default factor if no surprise is available

            # Example: Only forget if surprise factor is below a certain threshold
            if surprise_factor < surprise_threshold:
                # Reset the node's memory content
                node.memory_chunk = torch.zeros_like(node.memory_chunk)
                node.centroid = torch.zeros_like(node.centroid)
                node.timestamp = time.time()

                # Reset the corresponding surprise node's content
                if surprise_node is not None:
                    surprise_node.surprise_chunk = torch.zeros_like(surprise_node.surprise_chunk)

            # Recursively forget in the children
            for child, surprise_child in zip(node.children, surprise_node.children):
                self._forget_word_recursive(child, surprise_child, word_vector, similarity_threshold, surprise_threshold)
        else:
            # Recursively check the children
            for child, surprise_child in zip(node.children, surprise_node.children):
                self._forget_word_recursive(child, surprise_child, word_vector, similarity_threshold, surprise_threshold)

    def _calculate_surprise_factor(self, surprise_chunk):
        """
        Calculates a factor based on the surprise_chunk that modulates the forgetting rate.
        """
        base_factor = 1.0
        surprise_influence = 0.5
        surprise_value = surprise_chunk.item()
        factor = base_factor + surprise_influence * np.exp(surprise_value)
        return factor

    def _get_word_vector(self, word):
        """
        Converts a word to its vector representation (embedding).
        """
        return torch.randn(self.root_memory_chunk_size[0], dtype=torch.float32)

    def byte_to_patch(self, byte_sequence, layer_index):
        """
        Converts a byte sequence into a sequence of patch representations using the BLT Local Encoder.
        """
        # 1. Convert bytes to BLT input format (if necessary)
        blt_input = self._prepare_blt_input(byte_sequence)

        # 2. Encode the byte sequence using the BLT Local Encoder
        with torch.no_grad():
            patch_representations = self.blt_model.local_encoder(blt_input)

        # 3. Determine patch boundaries based on your chosen patching strategy
        patch_boundaries = self._determine_patch_boundaries(patch_representations, layer_index)

        # 4. Group the patch representations into patches
        patches = self._group_into_patches(patch_representations, patch_boundaries)

        return patches

    def _prepare_blt_input(self, byte_sequence):
        """
        Prepares a byte sequence for input to the BLT Local Encoder.
        """
        # Example: Add a start-of-sequence token (e.g., 0)
        return torch.cat([torch.tensor([0], dtype=torch.uint8), byte_sequence])

    def _determine_patch_boundaries(self, patch_representations, layer_index):
        """
        Determines patch boundaries based on your chosen patching strategy.
        """
        # Example: Using entropy-based patching with a global threshold
        entropy_threshold = self.entropy_thresholds[layer_index]
        patch_boundaries = []
        for i in range(len(patch_representations)):
            # Calculate entropy for the current patch representation
            entropy = self._calculate_entropy(patch_representations[i])

            if entropy > entropy_threshold:
                patch_boundaries.append(i)

        return patch_boundaries

    def _group_into_patches(self, patch_representations, patch_boundaries):
        """
        Groups the patch representations into patches based on the determined boundaries.
        """
        patches = []
        start = 0
        for end in patch_boundaries:
            patches.append(patch_representations[start:end+1])
            start = end + 1
        if start < len(patch_representations):
            patches.append(patch_representations[start:])
        return patches

    def _calculate_entropy(self, patch_representation):
        """
        Calculates the entropy of a patch representation.
        """
        return torch.randn(1).item()

    def calculate_surprise(self, patch, context_patches):
        """
        Calculates the surprise value for a given patch based on the context.
        """
        # 1. Convert the patch to BLT input format (if necessary)
        blt_input = self._prepare_blt_input(patch)

        # 2. Use the BLT Local Decoder to get the probability distribution over the next patch
        with torch.no_grad():
            next_patch_probabilities = self.blt_model.local_decoder.predict_next_patch(context_patches, blt_input)

        # 3. Convert the distribution over byte sequences (patches) into a single surprise value.
        surprise = -torch.log(next_patch_probabilities[actual_next_patch])

        return surprise.item()

    def _get_context_patches(self, node):
        """
        Extracts context patches for a given node.
        """
        context_patches = []
        current_node = node
        
        # Traverse up the hierarchy to gather context from parent nodes
        while current_node.parent is not None and len(context_patches) < self.max_context_patches:
            context_patches.insert(0, current_node.parent.centroid)  # Assuming centroid can represent the patch
            current_node = current_node.parent

        return context_patches[:self.max_context_patches]

    def trigger_memory_optimization(self, similarity_threshold, time_threshold, max_chunk_size=500, available_time_seconds=None):
        """
        Triggers the memory optimization process manually.
        """
        if available_time_seconds is None:
            available_time_seconds = float('inf')

        self.merge_similar_nodes_chunked(similarity_threshold, available_time_seconds, max_chunk_size)
        self.check_cross_layer_similarity_chunked(similarity_threshold, time_threshold, max_chunk_size=max_chunk_size, available_time_seconds=available_time_seconds)
        self.save("memory_state.pickle")

    def _schedule_memory_optimization(self, similarity_threshold, time_threshold, max_chunk_size=500, interval_hours=4):
        """
        Schedules the memory optimization process to run automatically at a fixed interval.
        """
        def optimization_task():
            try:
                print(f"Starting memory optimization at {time.ctime()}")
                available_time_seconds = interval_hours * 3600 * 0.8
                self.trigger_memory_optimization(similarity_threshold, time_threshold, max_chunk_size, available_time_seconds)
                print(f"Finished memory optimization at {time.ctime()}")
            finally:
                timer = threading.Timer(interval_hours * 3600, optimization_task)
                timer.daemon = True
                timer.start()

        optimization_task()


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
