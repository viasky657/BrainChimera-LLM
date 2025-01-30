
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from NueralMemoryLayers import HierarchicalMemory

class BinaryLatentTransformer(nn.Module):
    """Transformer encoder with Multi-State RNN features, reflection, and episodic memory for latent processing"""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, max_states: Optional[int] = None, patch_size: int = 4, num_latent_states: int = 4, reflection_threshold: float = 0.5, state_history_size=5, initial_temperature: float = 1.0, temperature_decay: float = 0.995, min_temperature: float = 0.5, b_star_n_star: int = 4, memory_layer: Optional[HierarchicalMemory] = None):
        super().__init__()
        self.memory_layer = memory_layer if memory_layer is not None else HierarchicalMemory(
            num_layers=4,
            root_memory_chunk_size=(hidden_size,),
            cache_capacity=10000
        )
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_states = max_states
        self.patch_size = patch_size
        self.num_latent_states = num_latent_states
        self.reflection_threshold = reflection_threshold  # Threshold for determining if a state is low quality

        # Byte embedding layer
        self.byte_embedding = nn.Embedding(256, hidden_size)

        # Patch encoder
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_size * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # State compression policy (TOVA) - Optional
        self.compression_enabled = max_states is not None
        if self.compression_enabled:
            self.tova_query = nn.Linear(hidden_size, hidden_size)
            self.tova_key = nn.Linear(hidden_size, hidden_size)
            self.tova_value = nn.Linear(hidden_size, hidden_size)

        # Latent mode flag
        self.latent_mode = False

        # Thought conditioning flag
        self.thought_conditioning = False

        # RNN-like update for latent states in latent mode
        self.latent_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # State selection/combination mechanism (e.g., attention)
        self.state_selector = nn.Linear(hidden_size, 1)

        self.num_latent_states = num_latent_states
        self.state_history_size = state_history_size  # Number of previous states to consider
        # State quality evaluator (now an RNN or Transformer)
        self.state_evaluator = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.state_evaluator_fc = nn.Linear(hidden_size, 1)  # Output the quality score

        # Buffer for storing previous states during latent processing
        self.state_history_buffer = []

        # B-STAR Integration
        self.temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.b_star_n_star = b_star_n_star #Value of n* in the balance score
        self.current_step = 0 #Keep track of the current training step
        self.adaptation_interval = 500 #How often to adjust temperature (based on paper)
        self.evaluation_set_size = 600 #How many samples to use for balance score evaluation

        # Placeholder for storing exploration data
        self.exploration_data = None

    def forward(self, x: torch.Tensor, thought_targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Store memories in the episodic memory layer
        if x is not None:
            self.memory_layer.process(x)
            
        # Access relevant memories from episodic memory
        # This can be used to condition the generation
        memory_output = self.memory_layer.retrieve()
        """
        Forward pass with Multi-State RNN features and reflection

        Args:
            x: Input tensor.
                - In language mode: [batch_size, seq_len] (bytes)
                - In latent mode: [batch_size, num_patches, num_latent_states, hidden_size]
            thought_targets: Optional tensor of target binary vectors [batch_size, num_patches, hidden_size]
            mask: Optional mask for the Transformer encoder

        Returns:
            output: Output tensor [batch_size, num_patches, hidden_size]
        """

        # --- Handle Input Based on Mode ---
        if self.latent_mode:
            # --- B-STAR Exploration (in Latent Mode) ---
            if self.exploration_data is not None and self.current_step % self.adaptation_interval == 0:
                self.adapt_temperature(self.exploration_data) # Adapt temperature every adaptation_interval steps
                self.exploration_data = None  # Reset exploration data

            # Multi-State Latent Mode:
            # 1. Recurrently update latent states
            # 2. Evaluate state quality (using history)
            # 3. Reflect (discard, rewind, or revise)
            if self.thought_conditioning and thought_targets is not None:
                # Combine each thought target with a latent state
                current_input = x + thought_targets.unsqueeze(2).repeat(1, 1, self.num_latent_states, 1)  # Or another combination function.
            else:
                current_input = x

            # Reshape to (batch_size * num_patches, num_latent_states, hidden_size) for RNN
            batch_size, num_patches, _, _ = current_input.shape
            current_input = current_input.view(batch_size * num_patches, self.num_latent_states, self.hidden_size)

            # RNN update with temperature sampling
            rnn_output, _ = self.latent_rnn(current_input)  # [batch_size * num_patches, num_latent_states, hidden_size]

            # State Evaluation (using history)

            # Update state history buffer
            self.state_history_buffer.append(rnn_output)
            if len(self.state_history_buffer) > self.state_history_size:
                self.state_history_buffer.pop(0)  # Remove oldest states if buffer is full

            # Concatenate current states with history
            state_history = torch.stack(self.state_history_buffer, dim=1)  # [batch_size * num_patches, history_size, num_latent_states, hidden_size]

            # Reshape for state evaluator
            state_history = state_history.view(batch_size * num_patches, -1, self.hidden_size)  # [batch_size * num_patches, history_size * num_latent_states, hidden_size]

            # Evaluate state quality
            state_eval_output, _ = self.state_evaluator(state_history)  # [batch_size * num_patches, num_latent_states, hidden_size]
            state_qualities = self.state_evaluator_fc(state_eval_output).squeeze(-1)  # [batch_size * num_patches, num_latent_states]

            # Reflection (here, we'll just discard low-quality states based on a threshold)
            mask = state_qualities > self.reflection_threshold  # [batch_size * num_patches, num_latent_states]
            reflected_output = torch.where(mask.unsqueeze(-1), rnn_output, torch.zeros_like(rnn_output))

            # State selection/combination (using attention as an example)
            attn_scores = self.state_selector(reflected_output).squeeze(-1)  # [batch_size * num_patches, num_latent_states]

            # Apply temperature-scaled softmax
            attn_scores = attn_scores / self.temperature
            attn_weights = torch.softmax(attn_scores, dim=-1)

            selected_state = torch.sum(reflected_output * attn_weights.unsqueeze(-1), dim=1)  # [batch_size * num_patches, hidden_size]

            # Reshape back to [batch_size, num_patches, hidden_size]
            output = selected_state.view(batch_size, num_patches, self.hidden_size)
        else:
            # Language Mode: Convert bytes to patches
            current_input = self.bytes_to_patches(x)  # [batch_size, num_patches, hidden_size]

            # Initialize multiple latent states per patch
            current_input = current_input.unsqueeze(2).repeat(1, 1, self.num_latent_states, 1)  # [batch_size, num_patches, num_latent_states, hidden_size]

            # --- Transformer Encoder ---
            # Flatten the num_patches and num_latent_states dimensions for Transformer input
            batch_size, num_patches, _, _ = current_input.shape
            current_input = current_input.view(batch_size, num_patches * self.num_latent_states, self.hidden_size)

            output = self.transformer_encoder(current_input, mask=mask)

            # Reshape back to [batch_size, num_patches, num_latent_states, hidden_size]
            output = output.view(batch_size, num_patches, self.num_latent_states, self.hidden_size)

        # --- Apply TOVA compression (Optional) ---
        if self.compression_enabled and self.latent_mode and self.max_states is not None:
            output = self._tova_compress(output)

        # --- B-STAR Update Step Counter ---
        self.current_step += 1

        return output

    def _tova_compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress states using a simplified TOVA-like policy."""
        batch_size, num_patches, num_latent_states, hidden_size = h.shape

        # 1. Pre-compute Queries, Keys, and Values
        query = self.tova_query(h)  # [batch_size, num_patches, num_latent_states, hidden_size]
        key = self.tova_key(h)      # [batch_size, num_patches, num_latent_states, hidden_size]
        value = self.tova_value(h)  # [batch_size, num_patches, num_latent_states, hidden_size]

        # 2. Attention per Patch (Intra-Patch Attention)
        # This reduces computations by computing attention within each patch first.
        attn_scores_intra = torch.einsum("bpqh,bpkh->bpqk", query, key) / (hidden_size ** 0.5)  # [batch_size, num_patches, num_latent_states, num_latent_states]
        attn_weights_intra = torch.softmax(attn_scores_intra, dim=-1)
        attended_states_intra = torch.einsum("bpqk,bpkh->bpqh", attn_weights_intra, value)  # [batch_size, num_patches, num_latent_states, hidden_size]

        # 3. Reduce Latent States (within each patch)
        # We'll keep only the top-k states based on intra-patch attention.
        # This further reduces memory before inter-patch attention.
        k_intra = min(self.num_latent_states // 2, self.max_states) # Example: Keep at least half, but not more than max_states
        _, topk_indices_intra = torch.topk(attn_weights_intra.sum(dim=-2), k_intra, dim=-1) # Sum scores across states, then find top-k indices
        attended_states_reduced = torch.gather(attended_states_intra, 2, topk_indices_intra.unsqueeze(-1).expand(-1, -1, -1, hidden_size)) # [batch_size, num_patches, k_intra, hidden_size]

        # 4. Inter-Patch Attention (Global Attention)
        # Now, compute attention across patches, but using the reduced set of states.
        # Reshape to combine num_patches and reduced latent states for global attention
        attended_states_reduced = attended_states_reduced.view(batch_size, num_patches * k_intra, hidden_size)
        query_global = self.tova_query(attended_states_reduced) # [batch_size, num_patches * k_intra, hidden_size]
        key_global = self.tova_key(attended_states_reduced) # [batch_size, num_patches * k_intra, hidden_size]
        value_global = self.tova_value(attended_states_reduced) # [batch_size, num_patches * k_intra, hidden_size]

        attn_scores_global = torch.einsum("bqi,bki->bqk", query_global, key_global) / (hidden_size ** 0.5)  # [batch_size, num_patches * k_intra, num_patches * k_intra]

        # 5. Masked Attention (Optional but Recommended)
        # Create a mask to prevent attending to states from the same patch.
        # This encourages diversity in the selected states.
        if num_patches > 1:  # Only apply masking if there's more than one patch
            mask_global = torch.block_diag(*[torch.ones(k_intra, k_intra) for _ in range(num_patches)]).to(attn_scores_global.device)
            attn_scores_global = attn_scores_global.masked_fill(mask_global == 1, float('-inf'))

        attn_weights_global = torch.softmax(attn_scores_global, dim=-1) # [batch_size, num_patches * k_intra, num_patches * k_intra]

        # 6. Select Top-M States (Global)
        # Select the top states globally, across all patches.
        _, topk_indices_global = torch.topk(attn_weights_global.sum(dim=-2), self.max_states, dim=-1)  # Sum scores across patches to rank states globally, then find top-k indices
        attended_states_global = torch.gather(value_global, 1, topk_indices_global.unsqueeze(-1).expand(-1, -1, hidden_size)) # [batch_size, max_states, hidden_size]

        # 7. Combine Selected States (Optional)
        # You can optionally combine the globally selected states using a weighted average
        # based on their global attention scores. This step is not strictly necessary
        # if you want to keep the states separate.
        attn_weights_final = torch.gather(attn_weights_global.sum(dim=-2), 1, topk_indices_global)
        attn_weights_final = attn_weights_final / attn_weights_final.sum(dim=-1, keepdim=True)  # Normalize
        combined_state = torch.einsum("bmi,bm->bi", attended_states_global, attn_weights_final) # [batch_size, hidden_size]

        # 8. Reshape and Return
        # If you combined the states, expand the dimensions to match the original shape
        # If you didn't combine them, attended_states_global is already in the correct shape
        if num_patches > 1:
            combined_state = combined_state.unsqueeze(1).unsqueeze(2).expand(-1, num_patches, self.num_latent_states, -1) # Expand to match original shape
        else:
            combined_state = combined_state.unsqueeze(1).expand(-1, num_patches, -1) # Expand to match original shape
            combined_state = combined_state.unsqueeze(2).repeat(1, 1, self.num_latent_states, 1) # Repeat to match original shape for num_latent_states
        return combined_state

    def bytes_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a sequence of bytes to a sequence of patches.

        Args:
            x: Input tensor of bytes [batch_size, seq_len]

        Returns:
            A tensor of patches [batch_size, num_patches, hidden_size]
        """
        batch_size, seq_len = x.shape
        num_patches = seq_len // self.patch_size

        # Convert bytes to embeddings
        embeddings = self.byte_embedding(x.long())  # [batch_size, seq_len, hidden_size]

        # Reshape into patches
        embeddings = embeddings.view(batch_size, num_patches, self.patch_size, self.hidden_size)

        # Concatenate embeddings within each patch
        concatenated = embeddings.view(batch_size, num_patches, -1)  # [batch_size, num_patches, patch_size * hidden_size]

        # Process each patch through the patch encoder
        latent_patches = self.patch_encoder(concatenated)  # [batch_size, num_patches, hidden_size]

        return latent_patches

    def enable_latent_mode(self):
        """Switch to latent mode"""
        self.latent_mode = True
        self.state_history_buffer = []  # Clear state history

    def disable_latent_mode(self):
        """Switch to language mode"""
        self.latent_mode = False

    def enable_thought_conditioning(self):
        """Enable thought conditioning"""
        self.thought_conditioning = True

    def disable_thought_conditioning(self):
        """Disable thought conditioning"""
        self.thought_conditioning = False

    def adapt_temperature(self, exploration_data: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        """
        Adapts the temperature based on the B-STAR balance score.

        Args:
            exploration_data: A list of tuples, where each tuple contains:
                - states: The latent states [batch_size, num_patches, num_latent_states, hidden_size]
                - state_qualities: The quality scores for each state [batch_size * num_patches, num_latent_states]
                - rewards: The rewards for each sample (e.g., from a reward model) [batch_size * num_patches]
        """

        best_temperature = self.temperature
        best_balance_score = -float('inf')

        # Grid search over a range of temperatures
        for temperature in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            total_balance_score = 0

            for states, state_qualities, rewards in exploration_data:
                # Select states based on temperature-scaled attention
                batch_size, num_patches, num_latent_states, hidden_size = states.shape
                states = states.view(batch_size * num_patches, num_latent_states, hidden_size)

                attn_scores = self.state_selector(states).squeeze(-1)  # [batch_size * num_patches, num_latent_states]
                attn_scores = attn_scores / temperature
                attn_weights = torch.softmax(attn_scores, dim=-1)
                selected_states = torch.sum(states * attn_weights.unsqueeze(-1), dim=1)  # [batch_size * num_patches, hidden_size]

                # Calculate balance score for each sample
                for i in range(batch_size * num_patches):
                    #Get the reward for the current sample
                    reward = rewards[i]

                    #Count unique correct states (assuming reward > threshold means correct)
                    unique_correct_count = 0
                    seen_states = set()
                    for j in range(self.num_latent_states):
                      if reward > 0 and tuple(selected_states[i, :].tolist()) not in seen_states: # You might need a better way to check for uniqueness
                        unique_correct_count += 1
                        seen_states.add(tuple(selected_states[i, :].tolist()))

                    #Count selected states (based on some criteria, e.g., above a threshold)
                    selected_count = 0
                    for j in range(self.num_latent_states):
                      if attn_weights[i, j] > 0.1:  # Example threshold
                        selected_count += 1

                    #Calculate balance score
                    balance_score = min(unique_correct_count / self.b_star_n_star, 1) * (min(unique_correct_count, selected_count) / selected_count if selected_count > 0 else 0)
                    total_balance_score += balance_score

            # Average balance score for this temperature
            avg_balance_score = total_balance_score / len(exploration_data)

            # Update best temperature if needed
            if avg_balance_score > best_balance_score:
                best_balance_score = avg_balance_score
                best_temperature = temperature

        # Update temperature (with decay)
        self.temperature = max(best_temperature * self.temperature_decay, self.min_temperature)
        print(f"Adapted temperature to: {self.temperature:.3f}")

    def add_to_exploration_data(self, states: torch.Tensor, state_qualities: torch.Tensor, rewards: torch.Tensor):
        """Adds data to the exploration buffer for temperature adaptation."""
        if self.exploration_data is None:
            self.exploration_data = []
        
        # Add to current batch
        self.exploration_batch.append((states.detach().cpu(), state_qualities.detach().cpu(), rewards.detach().cpu()))
        
        # Process batch if full
        if len(self.exploration_batch) >= self.exploration_batch_size:
            self._process_exploration_batch()
            
        # Keep exploration data within size limit
        if len(self.exploration_data) >= self.evaluation_set_size:
            self.exploration_data = self.exploration_data[-self.evaluation_set_size:]
            
    def _process_exploration_batch(self):
        """Processes a batch of exploration data."""
        if not self.exploration_batch:
            return

        # Combine batch data
        states_list, qualities_list, rewards_list = zip(*self.exploration_batch)
        states = torch.stack(states_list)
        qualities = torch.stack(qualities_list)
        rewards = torch.stack(rewards_list)

        # Calculate quality metrics
        quality_mask = qualities > self.state_quality_threshold
        unique_states = torch.unique(states[quality_mask], dim=0)
        num_unique = len(unique_states)

        # Calculate diversity metric
        diversity = torch.unique(rewards).size(0) / rewards.size(0)

        # Add to exploration data
        self.exploration_data.append((states, qualities, rewards, num_unique, diversity))
        self.exploration_batch = []

'''

**Key Changes and Explanations:**

1. **B-STAR Parameters:**
    *   `initial_temperature`, `temperature_decay`, `min_temperature`: Control the temperature adaptation process.
    *   `b_star_n_star`:  Corresponds to the `n*` parameter in the balance score formula, representing the desired number of correct responses per query.
    *   `current_step`: Tracks the training step for periodic temperature adaptation.
    *   `adaptation_interval`: Determines how many steps to take before adapting the temperature.
    *   `evaluation_set_size`:  Specifies the number of data points to collect for evaluating the balance score and adapting the temperature.
    *   `exploration_data`: A list to store data used for temperature adaptation.

2. **`adapt_temperature()` Method:**
    *   This method implements the core logic of B-STAR's temperature adaptation.
    *   It performs a grid search over a predefined set of temperatures (you can customize this set).
    *   For each temperature:
        *   It re-calculates the attention weights using the current `exploration_data` and the candidate temperature.
        *   It computes the balance score for each sample in the `exploration_data`. The balance score calculation will need to be adjusted according to your specific definition of "correct" and "selected" states. I've provided a basic example, but you'll need to refine it.
        *   It averages the balance scores across the `exploration_data`.
    *   It selects the temperature that yields the highest average balance score.
    *   It updates the `self.temperature` using the best temperature, applying the decay factor and ensuring it doesn't fall below `min_temperature`.
    *   The balance score is a crucial element of B-STAR. The way I've implemented it here is a placeholder and needs to be tailored to your specific task and how you define the quality of generated responses. The formula from the paper is:

        ```
        balance_score = min(n_i' / n*, 1) * (n_i / n_i)
        ```

        Where:
        *   `n_i'` is the number of unique, correct responses.
        *   `n_i` is the total number of selected responses.
        *   `n*` is a hyperparameter ( `b_star_n_star` in the code).

3. **`add_to_exploration_data()` Method:**
    *   This is a helper method to accumulate data that will be used later by `adapt_temperature()`. You'll need to call this method during training in latent mode to store the relevant information (states, state qualities, and rewards).

4. **Integration into `forward()`:**
    *   The `forward()` method now includes a check: `if self.exploration_data is not None and self.current_step % self.adaptation_interval == 0:`. This means temperature adaptation happens only in latent mode and only at intervals defined by `adaptation_interval`.
    *   The line `self.exploration_data = None` resets the buffer after adaptation.
    *   The temperature is used in the softmax calculation for attention weights: `attn_scores = attn_scores / self.temperature`.
    *   `self.current_step` is incremented at the end of each forward pass.

**How to Use:**

1. **Initialization:** Set the B-STAR parameters when creating an instance of `BinaryLatentTransformer`.
2. **Training Loop (in latent mode):**
    *   Call `model.enable_latent_mode()` before entering the latent processing phase.
    *   During training in latent mode, periodically (e.g., after each batch or epoch), collect the necessary data:
        *   `states`: The output of the RNN (before selection).
        *   `state_qualities`: The output of the state evaluator.
        *   `rewards`:  The rewards obtained for each sample (you'll need to define how these are calculated based on your task).
        *   Call `model.add_to_exploration_data(states, state_qualities, rewards)` to store this data.
    *   The `forward()` method will automatically trigger temperature adaptation at the specified intervals.
3. **Reward Mechanism:** You'll need a way to get rewards during training. This could be a separate reward model (as in the paper) or some other form of feedback that indicates the quality of the generated responses.
4. **Balance Score Refinement:** The balance score calculation in `adapt_temperature()` needs to be carefully adjusted to match your specific task and the way you evaluate the correctness and selection of latent states.

**Important Considerations:**

*   **Computational Cost:** The temperature adaptation process does add some computational overhead because of the grid search and balance score calculations. However, it's done periodically, not at every step.
*   **Hyperparameter Tuning:** The B-STAR parameters themselves might need some tuning (e.g., `initial_temperature`, `temperature_decay`, `min_temperature`, `adaptation_interval`, `evaluation_set_size`, `b_star_n_star`).
*   **Exploration Data Size:** The `evaluation_set_size` determines how much data is used for temperature adaptation. You might need to experiment to find a good balance between adaptation accuracy and computational cost.

This revised response provides a more complete and detailed implementation of B-STAR's exploration policy and temperature adaptation within your `BinaryLatentTransformer`. Remember to adapt the balance score calculation and the reward mechanism to your specific needs. '''
