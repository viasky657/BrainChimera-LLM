
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode

# Custom Transformer encoder with attention tracking
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        self.attention_weights = attention_weights
        return output

    def get_attention_weights(self):
        return self.attention_weights

class SensoryPerception(nn.Module): #For predicting the (this) model's action impact on the environment.
    def __init__(self, input_channels: int, hidden_size: int):
        super().__init__()
        # Example: Simple CNN for image input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, hidden_size) # Assuming input images are resized to 32x32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_channels, height, width]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x # [batch_size, hidden_size]


class BinaryLatentTransformer(nn.Module):
    """Transformer encoder with Multi-State RNN features, reflection, episodic memory, and self-verification for latent processing"""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, max_states: Optional[int] = None, patch_size: int = 4, num_latent_states: int = 4, 
                 reflection_threshold: float = 0.5, state_history_size=5, initial_temperature: float = 1.0, temperature_decay: float = 0.995,
                   min_temperature: float = 0.5, b_star_n_star: int = 4, memory_layer: Optional[HierarchicalMemory] = None, self_criticism_layers: int = 2, 
                   self_criticism_hidden: int = 128, surprise_threshold: float=0.5, memory_influence_factor: float=0.5, state_quality_threshold: float=0.5, 
                   belief_state_size: int = 16, truthfulness_state_size: int = 8):
        super().__init__()
        self.memory_layer = memory_layer if memory_layer is not None else HierarchicalMemory(
            num_layers=4,
            root_memory_chunk_size=(hidden_size,),
            cache_capacity=10000
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_states = max_states
        self.patch_size = patch_size
        self.num_latent_states = num_latent_states
        self.reflection_threshold = reflection_threshold  # Threshold for determining if a state is low quality
        # Self-criticism components
        self.self_criticism_layers = self_criticism_layers
        self.self_criticism_hidden = self_criticism_hidden
        self.self_criticism_layer = nn.GRU(
            input_size=hidden_size,
            hidden_size=self_criticism_hidden,
            num_layers=self.self_criticism_layers,
            batch_first=True
        )
        self.quality_predictor = nn.Linear(self_criticism_hidden, 1)

        # Byte embedding layer
        self.byte_embedding = nn.Embedding(256, hidden_size)

        # Patch encoder
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_size * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        encoder_layer = CustomTransformerEncoderLayer(
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

       # Latent RNN now takes additional inputs for belief and truthfulness
        self.latent_rnn = nn.GRU(hidden_size + belief_state_size + truthfulness_state_size, hidden_size, batch_first=True)

        # State selection/combination mechanism (e.g., attention)
        self.state_selector = nn.Linear(hidden_size, 1)

        self.state_history_size = state_history_size  # Number of previous states to consider
        # State evaluator now takes additional inputs for belief and truthfulness
        self.state_evaluator = nn.GRU(
            input_size=hidden_size + belief_state_size + truthfulness_state_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.state_evaluator_fc = nn.Linear(hidden_size, 1)  # Output the quality score

        # Internal reward mechanism
        self.reward_generator = nn.Linear(hidden_size, 1)

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
        self.exploration_batch_size = 10 # Example batch size, adjust as needed
        self.exploration_batch = []  # Initialize the batch
        
        #Hyperparameters for memory and surprise
        self.surprise_threshold = surprise_threshold
        self.memory_influence_factor = memory_influence_factor
        self.state_quality_threshold = state_quality_threshold
        
        # Placeholder for storing exploration data
        self.exploration_data = None

        # Empathy Truthfulness belief about self and internal states versus external actions. Correction on untruthful behavior. 
        self.belief_state_size = belief_state_size
        self.truthfulness_state_size = truthfulness_state_size

        # Initialize belief and truthfulness states
        self.initial_belief_state = nn.Parameter(torch.zeros(1, 1, 1, belief_state_size))
        self.initial_truthfulness_state = nn.Parameter(torch.zeros(1, 1, 1, truthfulness_state_size))

    def calculate_surprise(self, current_input: torch.Tensor, memory_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate surprise factor based on gradient, memory comparison, and hierarchical context"""
        # Calculate gradient-based surprise
        gradient = torch.autograd.grad(self.loss, current_input, retain_graph=True)[0]
        gradient_magnitude = torch.norm(gradient, p=2, dim=-1)  # [batch_size, seq_len]

        # Calculate memory-based surprise
        similarity = torch.cosine_similarity(current_input, memory_output, dim=-1)  # [batch_size, seq_len]
        similarity = similarity.mean(dim=1)  # [batch_size]

        # Calculate hierarchical context surprise
        context_surprise = self.memory_layer.calculate_context_surprise(current_input, memory_output)

        # Combine surprise factors
        gradient_surprise = gradient_magnitude.mean(dim=1)
        memory_surprise = 1 - similarity
        combined_surprise = gradient_surprise * memory_surprise * context_surprise  # [batch_size]

        return gradient_surprise, memory_surprise, combined_surprise

    def handle_forgetting(self, gradient_surprise: torch.Tensor, memory_surprise: torch.Tensor, combined_surprise: torch.Tensor):
        """Handle memory forgetting based on surprise factors"""
        # Forget outdated memories
        if combined_surprise.mean() > self.surprise_threshold:
            self.memory_layer.forget_words(["outdated"], 0.6, combined_surprise.mean().item())

        # Prune low-confidence nodes by setting weights to zero (not removing)
        self.memory_layer.prune_children_zero_weights(
            self.memory_layer.memory_layers[self.memory_layer.active_layer], 
            self.memory_layer.active_layer,
            similarity_threshold=0.8,
            reconnection_threshold=0.7
        )

    def calculate_surprise_factor(self, memory_chunk: torch.Tensor) -> float:
        """Calculate surprise factor based on the memory chunk's surprise content"""
        return self.memory_layer._calculate_surprise_factor(memory_chunk)

    def forget_words(self, words_to_forget: List[str], similarity_threshold: float = 0.6, surprise_threshold: float = 0.5):
        """Forget specific memories based on words and surprise factors"""
        self.memory_layer.forget_words(words_to_forget, similarity_threshold, surprise_threshold)

    def prune_children_zero_weights(self, node: MemoryNode, layer_index: int, similarity_threshold: float = 0.8, reconnection_threshold: float = 0.7):
        """Prune children of a node based on similarity and reconnection thresholds by setting weights to zero instead of removing."""
        self.memory_layer.prune_children_zero_weights(node, layer_index, similarity_threshold, reconnection_threshold)

    def calculate_truthfulness_reward(self, predicted_state: torch.Tensor, ground_truth_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates a reward based on the consistency between the predicted state and the actual state.

        Args:
            predicted_state: The state predicted by the model [batch_size * num_patches, num_latent_states, hidden_size]
            ground_truth_state: The actual ground truth state [batch_size, seq_len] or [batch_size, num_patches, num_latent_states, hidden_size]

        Returns:
            A tensor representing the truthfulness reward for each latent state.
        """

        # --- 1. Prediction vs. Observation (for Language Mode Input) ---
        if not self.latent_mode:
            # Decode the predicted_state back into a sequence of bytes (you'll need to implement this based on your patch encoding)
            decoded_predictions = self.decode_to_bytes(predicted_state) # [batch_size, seq_len]

            # Calculate the difference between the decoded predictions and the ground truth
            # Example: Mean squared error (can also use cross-entropy if you one-hot encode the bytes)
            consistency = -torch.mean((decoded_predictions - ground_truth_state.float())**2, dim=-1) # [batch_size]

        # --- 2. Prediction vs. Observation (for Latent Mode Input) ---
        else:
            # Calculate the difference between the predicted_state and the ground_truth_state
            consistency = -torch.mean((predicted_state - ground_truth_state)**2, dim=-1) # [batch_size * num_patches, num_latent_states]

        # --- 3. Internal Consistency (Optional) ---
        # You could also add a term that measures the consistency between different latent states
        # or between the latent states and the agent's belief state.
        # Example:
        # internal_consistency = -torch.var(predicted_state, dim=1) # Low variance across latent states might indicate higher confidence

        # Combine the consistency measures (you might need to weight them differently)
        truthfulness_reward = consistency # + internal_consistency

        return truthfulness_reward
    
    def decode_to_bytes(self, latent_states: torch.Tensor) -> torch.Tensor: #Need to change this to work with the embedding into the latent space (RNN network).
        """
        Decodes a sequence of latent states back into a sequence of bytes.

        Args:
            latent_states: Input tensor of latent states [batch_size * num_patches, num_latent_states, hidden_size]

        Returns:
            A tensor of decoded bytes [batch_size, seq_len]
        """
        batch_size_times_num_patches, num_latent_states, hidden_size = latent_states.shape
        batch_size = int(batch_size_times_num_patches**(1/2)) # Assumes a square number of patches
        # 1. Select/Combine Latent States
        # Here, we'll use a simple average across latent states.
        # You can replace this with a more sophisticated mechanism like attention if needed.
        combined_state = torch.mean(latent_states, dim=1) # [batch_size * num_patches, hidden_size]

        # 2. Reshape for Patch Decoding
        combined_state = combined_state.view(batch_size, -1, hidden_size) # [batch_size, num_patches, hidden_size]

        # 3. Invert the Patch Encoder
        # We need to reverse the operations done in `bytes_to_patches`.
        # This is highly dependent on your specific implementation.
        # Here's a placeholder assuming your patch encoder is a simple linear layer:
        patch_decoding = nn.Linear(hidden_size, self.patch_size * hidden_size).to(combined_state.device) # You might need multiple layers
        decoded_patches = patch_decoding(combined_state)  # [batch_size, num_patches, patch_size * hidden_size]

        # 4. Reshape to Byte Embeddings
        decoded_embeddings = decoded_patches.view(batch_size, -1, hidden_size) # [batch_size, seq_len, hidden_size]

        # 5. Invert the Byte Embedding
        # Again, this depends on your implementation.
        # If you used an embedding layer, you might need to find the closest byte embedding for each output vector.
        # Here's a placeholder using a linear layer to project back to byte space:
        byte_projection = nn.Linear(hidden_size, 256).to(decoded_embeddings.device) # Project to 256 possible byte values
        byte_logits = byte_projection(decoded_embeddings) # [batch_size, seq_len, 256]

        # 6. Convert to Bytes
        decoded_bytes = torch.argmax(byte_logits, dim=-1) # [batch_size, seq_len]

        return decoded_bytes

    def update_belief_states(self, belief_states: torch.Tensor, rnn_output: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Updates the belief states based on the current state and attention weights.

        Args:
            belief_states: The current belief states.
            rnn_output: The output of the latent RNN.
            attn_weights: The attention weights over latent states.

        Returns:
            The updated belief states.
        """
        # Example:
        # - Increase belief in dimensions that correspond to high-quality states
        # - Decrease belief in dimensions that correspond to low-quality states

        # Calculate a weighted average of the RNN output based on attention weights
        weighted_rnn_output = torch.sum(rnn_output * attn_weights.unsqueeze(-1), dim=1)  # [batch_size * num_patches, hidden_size]

        # Update the belief states based on the weighted average
        # (You might need to use a more sophisticated update rule here)
        belief_states = belief_states + 0.1 * (weighted_rnn_output.unsqueeze(2) - belief_states) # [batch_size, num_patches, num_latent_states, belief_state_size]

        return belief_states

    def update_truthfulness_states(self, truthfulness_states: torch.Tensor, truthfulness_reward: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Updates the truthfulness states based on the truthfulness reward.

        Args:
            truthfulness_states: The current truthfulness states.
            truthfulness_reward: The calculated truthfulness reward.
            attn_weights: The attention weights over latent states.

        Returns:
            The updated truthfulness states.
        """
        # Example:
        # - Increase truthfulness in dimensions that correspond to high truthfulness rewards
        # - Decrease truthfulness in dimensions that correspond to low truthfulness rewards

        # Scale the truthfulness reward based on attention weights
        scaled_truthfulness_reward = torch.sum(truthfulness_reward.unsqueeze(-1) * attn_weights, dim=1)  # [batch_size * num_patches]

        # Update the truthfulness states
        # (You might need to use a more sophisticated update rule here, like using a separate RNN)
        truthfulness_states = truthfulness_states + 0.1 * (scaled_truthfulness_reward.unsqueeze(-1).unsqueeze(-1) - truthfulness_states) # [batch_size, num_patches, num_latent_states, truthfulness_state_size]

        return truthfulness_states

    def should_rewind(self, state_qualities: torch.Tensor, truthfulness_reward: torch.Tensor) -> bool:
        """
        Determines whether the rewind mechanism should be triggered.

        Args:
            state_qualities: The quality scores of the current states.
            truthfulness_reward: The truthfulness reward.

        Returns:
            True if the rewind mechanism should be triggered, False otherwise.
        """
        # Example criteria:
        # 1. Low average state quality
        # 2. Low truthfulness reward
        # 3. High discrepancy between predicted and actual rewards (if available)

        return torch.mean(state_qualities) < self.reflection_threshold or torch.mean(truthfulness_reward) < 0 # Adjust thresholds as needed

    def rewind(self, reflected_output: torch.Tensor, belief_states: torch.Tensor, truthfulness_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rewinds the latent state to a previous point in the history and modifies belief/truthfulness.

        Args:
            reflected_output: The current reflected output.
            belief_states: The current belief states.
            truthfulness_states: The current truthfulness states.

        Returns:
            The reflected output, belief states, and truthfulness states after rewinding.
        """
        if not self.state_history_buffer:
            #If the buffer is empty then return the initial states
            return reflected_output, belief_states, truthfulness_states

        # 1. Select a Rewind Point
        # Example: Randomly choose a point from the history
        rewind_idx = torch.randint(0, len(self.state_history_buffer), (1,)).item()

        # 2. Overwrite with Rewound State
        # Example: Replace the current state with the state at the rewind point
        rewound_state = self.state_history_buffer[rewind_idx] # [batch_size * num_patches, num_latent_states, hidden_size]
        reflected_output = rewound_state

        # 3. Modify Belief/Truthfulness States
        # Example: Decrease confidence/truthfulness based on the rewind
        belief_states = belief_states * 0.9  # Reduce belief in capabilities
        truthfulness_states = truthfulness_states * 0.9  # Reduce truthfulness score

        # 4. Remove History After Rewind Point
        self.state_history_buffer = self.state_history_buffer[:rewind_idx + 1]

        return reflected_output, belief_states, truthfulness_states

    def forward(self, x: torch.Tensor, thought_targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, return_introspection: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with Episodic Memory, Truthfulness Training, Multi-State RNN, and Reflection.

        Args:
            x: Input tensor.
                - In language mode: [batch_size, seq_len] (bytes)
                - In latent mode: [batch_size, num_patches, num_latent_states, hidden_size]
            thought_targets: Optional tensor of target binary vectors [batch_size, num_patches, hidden_size]
            mask: Optional mask for the Transformer encoder
            return_introspection: Flag to return introspection data

        Returns:
            output: Output tensor [batch_size, num_patches, hidden_size]
            episode_memory_tensor: Optional tensor for episodic memory
            introspection_data (optional): Dictionary containing introspection information
        """

        # --- 1. Episodic Memory Processing ---
        if x is not None:
            self.memory_layer.process(x)
        memory_output = self.memory_layer.retrieve()

        # --- 2. Surprise Handling ---
        surprise_factors = self.calculate_surprise(x, memory_output)
        gradient_surprise, memory_surprise, combined_surprise = surprise_factors
        self.handle_forgetting(gradient_surprise, memory_surprise, combined_surprise)

        if combined_surprise.mean() > self.surprise_threshold:
            self.memory_layer.update_memory(x)
            x = x + memory_output * self.memory_influence_factor

        self.memory_layer.trigger_memory_optimization(0.7, 3600)
        surprise_factor_val = self.memory_layer._calculate_surprise_factor(x)
        if surprise_factor_val > 0.5:
            self.memory_layer.forget_words(["outdated"], 0.6, surprise_factor_val)

        # --- 3. Mode-Specific Processing (Latent or Language) ---
        batch_size = x.shape[0] if x is not None else 0

        if self.latent_mode:
            # --- 3.1 B-STAR Exploration (in Latent Mode) ---
            if self.exploration_data is not None and self.current_step % self.adaptation_interval == 0:
                self.adapt_temperature(self.exploration_data)
                self.exploration_data = None

            # --- 3.2 Multi-State Latent Processing ---
            if self.thought_conditioning and thought_targets is not None:
                current_input = x + thought_targets.unsqueeze(2).repeat(1, 1, self.num_latent_states, 1)
            else:
                current_input = x

            # --- 3.3 Incorporate Belief and Truthfulness States ---
            num_patches = current_input.shape[1]
            belief_states = self.initial_belief_state.repeat(batch_size, num_patches, self.num_latent_states, 1)
            truthfulness_states = self.initial_truthfulness_state.repeat(batch_size, num_patches, self.num_latent_states, 1)
            current_input = torch.cat([current_input, belief_states, truthfulness_states], dim=-1)
            current_input = current_input.view(batch_size * num_patches, self.num_latent_states, -1)

            # --- 3.4 RNN Update and Self-Criticism ---
            rnn_output, _ = self.latent_rnn(current_input)
            self_criticism_input = rnn_output.view(batch_size * num_patches, self.num_latent_states, -1)
            self_criticism_output, _ = self.self_criticism_layer(self_criticism_input)
            output_quality = self.quality_predictor(self_criticism_output[:, -1, :]).squeeze(-1)
            rewards = torch.sigmoid(output_quality) # Internal rewards based on output quality

            # --- 3.5 Exploration Data Update ---
            if self.exploration_data is not None:
                self.add_to_exploration_data(
                    states=rnn_output,
                    state_qualities=output_quality.unsqueeze(-1),
                    rewards=rewards.mean(dim=1)
                )

            # --- 3.6 State History and Evaluation ---
            self.state_history_buffer.append(rnn_output)
            if len(self.state_history_buffer) > self.state_history_size:
                self.state_history_buffer.pop(0)
            state_history = torch.stack(self.state_history_buffer, dim=1)
            belief_truthfulness_history = torch.cat([belief_states, truthfulness_states], dim=-1).repeat(1, 1, 1, self.state_history_size).permute(0, 3, 1, 2)
            state_history = torch.cat([state_history, belief_truthfulness_history], dim=-1)
            state_history = state_history.view(batch_size * num_patches, -1, self.hidden_size + self.belief_state_size + self.truthfulness_state_size)

            state_eval_output, _ = self.state_evaluator(state_history)
            state_qualities = self.state_evaluator_fc(state_eval_output).squeeze(-1)

            # --- 3.7 Truthfulness Reward ---
            truthfulness_reward = self.calculate_truthfulness_reward(rnn_output, x)

            # --- 3.8 Reflection and Rewind ---
            mask = state_qualities > self.reflection_threshold
            reflected_output = torch.where(mask.unsqueeze(-1), rnn_output, torch.zeros_like(rnn_output))

            if self.should_rewind(state_qualities, truthfulness_reward):
                reflected_output, belief_states, truthfulness_states = self.rewind(reflected_output, belief_states, truthfulness_states)
                current_input = torch.cat([reflected_output, belief_states, truthfulness_states], dim=-1)
                current_input = current_input.view(batch_size * num_patches, self.num_latent_states, -1)
                rnn_output, _ = self.latent_rnn(current_input) # Update after rewind

            # --- 3.9 State Selection ---
            attn_scores = self.state_selector(reflected_output).squeeze(-1)
            attn_scores = attn_scores / self.temperature
            attn_weights = torch.softmax(attn_scores, dim=-1)
            selected_state = torch.sum(reflected_output * attn_weights.unsqueeze(-1), dim=1)
            output = selected_state.view(batch_size, num_patches, self.hidden_size)

            # --- 3.10 Update Belief and Truthfulness States ---
            belief_states = self.update_belief_states(belief_states, rnn_output, attn_weights)
            truthfulness_states = self.update_truthfulness_states(truthfulness_states, truthfulness_reward, attn_weights)
        
        else: # Language Mode
            # --- 3.11 Language Mode Processing ---
            current_input = self.bytes_to_patches(x)
            current_input = current_input.unsqueeze(2).repeat(1, 1, self.num_latent_states, 1)
            batch_size, num_patches, _, _ = current_input.shape
            current_input = current_input.view(batch_size, num_patches * self.num_latent_states, self.hidden_size)
            output = self.transformer_encoder(current_input, mask=mask)
            output = output.view(batch_size, num_patches, self.num_latent_states, self.hidden_size)

        # --- 4. TOVA Compression (Optional) ---
        if self.compression_enabled and self.latent_mode and self.max_states is not None:
            output = self._tova_compress(output)

        # --- 5. B-STAR Step Update ---
        self.current_step += 1

        # --- 6. Episodic Memory Storage ---
        episode_memory = {
            "output": output,
            "state_qualities": state_qualities if self.latent_mode else None,
            "attn_weights": attn_weights if self.latent_mode else None,
            "truthfulness_reward": truthfulness_reward if self.latent_mode else None,
            "belief_states": belief_states if self.latent_mode else None,
            "truthfulness_states": truthfulness_states if self.latent_mode else None,
        }
        episode_memory_tensor = self.memory_layer.add_episode(episode_memory)

        # --- 7. Introspection Data ---
        introspection_data = {
            'temperature_dynamics': {
                'initial_temperature': self.initial_temperature,
                'current_temperature': self.temperature,
                'temperature_decay': self.temperature_decay
            },
            'model_simulations': {
                'layer_activations': [],
                'attention_patterns': []
            }
        }

        if self.latent_mode:
            for i, layer in enumerate(self.transformer_encoder.layers):
                layer_output = layer(x)
                introspection_data['model_simulations']['layer_activations'].append(layer_output.detach())
                x = layer_output
                
                if isinstance(layer, CustomTransformerEncoderLayer):
                    attention_weights = layer.get_attention_weights()
                    introspection_data['model_simulations']['attention_patterns'].append(attention_weights.detach())

            if hasattr(self, 'attention'):
                attention_output = self.attention(x)
                introspection_data['model_simulations']['attention_patterns'] = attention_output.detach()

        # --- 8. Return Values ---
        if return_introspection:
            return output, episode_memory_tensor, introspection_data
        else:
            return output, episode_memory_tensor

    def generate_explanations(self, output: torch.Tensor) -> List[str]:
        """Generate explanations for model decisions"""
        # Implement explanation generation logic here
        # For now, return simple explanations
        return ["High confidence in output quality"] * output.shape[0]

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
