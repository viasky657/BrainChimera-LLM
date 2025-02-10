import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode
from OrthoGradOptimizer import OrthoGrad, OrthoAdamW, OrthoSGD # Import OrthoGrad optimizer For Grokking Enhancements
from StableCELoss import stable_cross_entropy_loss # Import Stable Cross-Entropy Loss For Grokking Enhancements
import typing 
import matplotlib.pyplot as plt
import seaborn as sns


class BrainRegionWrapper(nn.Module):
    """
    Wraps region-specific encoders (Transformers) and manages forward pass through them.
    """
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, regions: List[str]):
        super().__init__()
        self.regions = regions
        self.region_encoders = nn.ModuleDict({
            region: nn.TransformerEncoder(
                CustomTransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True),
                num_layers=num_layers
            )
            for region in regions
        })

    def forward(self, current_input_reshaped: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through region-specific encoders.

        Args:
            current_input_reshaped: Reshaped input tensor [batch_size * num_patches, num_latent_states, hidden_size]

        Returns:
            Dictionary of region-specific encoder outputs.
        """
        region_outputs = {}
        for region_name, encoder in self.region_encoders.items():
            region_outputs[region_name] = encoder(current_input_reshaped)  # Process input through each region-specific encoder
        return region_outputs


class PrefrontalCortex(nn.Module):
    """
    Prefrontal Cortex module adapted for COCONUT to interface with continuous thought.
    """
    def __init__(self, hidden_size, num_layers=3, binary_latent_transformer=None):  # Removed patch_size
        super().__init__()
        self.hidden_size = hidden_size

        # Reference to the parent BinaryLatentTransformer
        self.binary_latent_transformer = binary_latent_transformer  # Renamed to be consistent

        # Safety monitoring module
        self.metacognitive = MetacognitiveModule(hidden_size, hidden_size)

        # No latent_proj needed as input is already in latent space (continuous thought)

    def forward(self, continuous_thought):  # Input is now continuous_thought
        """Process continuous thought through symbolic latent space with safety checks"""

        # Process through BinaryLatentTransformer's latent Transformer Encoder (continuous thought)
        # Continuous thought is already the output of the encoder, no need to pass it again.
        latent_states = continuous_thought  # Directly use continuous thought as latent state

        # Perform safety monitoring on hidden states (continuous thought)
        safety_report = self.metacognitive(latent_states, latent_states)  # Monitor continuous thought

        # Apply corrections if needed
        if safety_report['needs_reflection']:
            corrected_states = safety_report['corrected_state']
            # Store reflection in memory
            self.binary_latent_transformer.memory_layer.store(  # Renamed to be consistent
                corrected_states.detach(),
                tags=['safety_correction_latent']  # Tag for latent safety correction
            )
            return corrected_states

        return latent_states


class MetacognitiveModule(nn.Module):
    """
    Enhanced Metacognitive Module adapted for continuous thought.
    """
    def __init__(self, hidden_dim, memory_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        # Original monitor layers for safety - now monitoring continuous thought
        self.thought_monitor = nn.Linear(hidden_dim, 1)  # Monitor continuous thought
        self.memory_monitor = nn.Linear(memory_dim, 1)

        # Reflection generation layers - operate on continuous thought
        self.reflection_net = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Error detection - detects errors in continuous thought
        self.error_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Self-correction mechanism - corrects continuous thought
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Memory of past reflections (stores last k reflections) - unchanged
        self.reflection_memory = []
        self.max_reflections = 5

    def forward(self, continuous_thought, memory):  # Input is now continuous_thought
        # Safety monitoring - monitoring continuous thought
        thought_score = torch.sigmoid(self.thought_monitor(continuous_thought))  # Monitor continuous thought
        memory_score = torch.sigmoid(self.memory_monitor(memory))
        safety_flag = (thought_score + memory_score) / 2

        # Generate reflection - based on continuous thought
        combined = torch.cat([continuous_thought, memory], dim=-1)
        reflection = self.reflection_net(combined)

        # Detect potential errors - in continuous thought
        error_prob = self.error_detector(reflection)

        # Store reflection in memory - unchanged
        if len(self.reflection_memory) >= self.max_reflections:
            self.reflection_memory.pop(0)
        self.reflection_memory.append(reflection.detach())

        # If error probability is high, attempt self-correction - correct continuous thought
        corrected_state = continuous_thought
        if error_prob > 0.5:
            # Use reflection and original state (continuous thought) for correction
            correction_input = torch.cat([continuous_thought, reflection], dim=-1)
            corrected_state = self.correction_net(correction_input)

        return {
            'safety_flag': safety_flag,
            'reflection': reflection,
            'error_prob': error_prob,
            'corrected_state': corrected_state,
            'needs_reflection': error_prob > 0.5
        }


class Value(nn.Module):
    """
    Value network for evaluating safety of continuous thought in latent space
    """
    def __init__(self, hidden_dim):  # Removed patch_size - not relevant for continuous thought
        super().__init__()
        self.hidden_dim = hidden_dim

        # Process continuous thought structure - directly linear layers on hidden_dim
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # Output single value for continuous thought quality
        )

    def forward(self, continuous_thought):  # Input is now continuous_thought
        """Compute safety values for continuous thought [batch_size * num_patches, num_latent_states, hidden_dim]"""
        # No flattening needed, input is already continuous thought
        values = torch.sigmoid(self.value_net(continuous_thought))  # Evaluate continuous thought directly
        return values  # Returns quality value of continuous thought


# Custom Transformer encoder with attention tracking
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None
        # Contribution utility tracking
        self.contribution_utility = None
        self.age = nn.Parameter(torch.tensor(0), requires_grad=False) # Age parameter, not trainable

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        self.attention_weights = attention_weights
        return output

    def get_attention_weights(self):
        return self.attention_weights

    def update_utility(self, input_tensor, output_tensor):
        """Update contribution utility based on input and output tensors."""
        if self.contribution_utility is None:
            self.contribution_utility = torch.zeros_like(self.weight).float() # Initialize if None
        instantaneous_contribution = torch.abs(self.weight) * torch.abs(output_tensor) # Example utility measure
        decay_rate = 0.99 # Decay rate for running average
        self.contribution_utility = decay_rate * self.contribution_utility + (1 - decay_rate) * instantaneous_contribution

    def increment_age(self):
        """Increment the age of the layer."""
        self.age += 1

    def get_utility(self):
        """Return the current contribution utility."""
        return self.contribution_utility

    def get_age(self):
        """Return the current age of the layer."""
        return self.age.item()


class SensoryPerception(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int):
        super().__init__()
        # Example: Simple CNN for image input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, hidden_size)  # Assuming input images are resized to 32x32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_channels, height, width]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x  # [batch_size, hidden_size]

class PolicyNetwork(nn.Module):  # Environmental impact, self-preservation, altruism, goal, and emotional well-being balancing.
    """
    Policy Network to dynamically adjust reward weights in BinaryLatentTransformer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.policy_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # Output weights should be in range [0, 1] - using Sigmoid activation
        )

    def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Policy Network.

        Args:
            policy_input: Input tensor for policy network [batch_size, input_size]

        Returns:
            Output tensor of reward weights [batch_size, output_size]
        """
        return self.policy_net(policy_input)


class OtherAgentPredictor(nn.Module):
    """RNN-based predictor for other agent's behavior, internal state, knowledge, and beliefs, adapted for COCONUT."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, belief_state_size: int,
                 truthfulness_state_size: int, max_belief_depth: int = 3, communication_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_belief_depth = max_belief_depth
        self.communication_size = communication_size
        # --- Emotion Prediction ---
        self.negative_emotion_predictor_other = nn.Linear(hidden_size, 1)  # Predict negative emotion of other
        self.negative_emotion_predictor_self = nn.Linear(hidden_size, 1)  # Predict negative emotion of self

        # RNN for observable behavior prediction - now predicts based on continuous thought
        self.behavior_rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # Input is continuous thought size

        # RNN for internal state prediction (if available) - now predicts based on continuous thought
        self.internal_state_rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers, batch_first=True)  # Input is continuous thought size + hidden size

        # --- Theory of Mind (Nested Beliefs) --- - unchanged
        self.belief_rnns = nn.ModuleList([
            nn.GRU(hidden_size if i == 0 else belief_state_size, belief_state_size, num_layers, batch_first=True)
            for i in range(max_belief_depth)
        ])

        # RNN for truthfulness state prediction - now predicts based on continuous thought and belief
        self.truthfulness_rnn = nn.GRU(hidden_size + belief_state_size, truthfulness_state_size, num_layers, batch_first=True)  # Input is continuous thought size + belief state size

        # Output layers for different aspects - unchanged, but operating on continuous thought representations
        self.action_predictor = nn.Linear(hidden_size, input_size)
        self.internal_state_predictor = nn.Linear(hidden_size, hidden_size)
        self.belief_predictors = nn.ModuleList([
            nn.Linear(belief_state_size, belief_state_size)
            for _ in range(max_belief_depth)
        ])
        self.truthfulness_predictor = nn.Linear(truthfulness_state_size, truthfulness_state_size)

        # Latent space for predictions - unchanged, but operating on continuous thought representations
        self.latent_space_behavior = nn.Linear(hidden_size, hidden_size // 2)
        self.latent_space_internal = nn.Linear(hidden_size, hidden_size // 2)
        self.latent_space_belief = nn.Linear(belief_state_size, belief_state_size // 2)
        self.latent_space_truthfulness = nn.Linear(truthfulness_state_size, truthfulness_state_size // 2)

        # --- Communication --- - unchanged
        self.communication_encoder = nn.Linear(hidden_size + belief_state_size + truthfulness_state_size, communication_size)
        self.communication_decoder = nn.Linear(communication_size, hidden_size)  # To influence the agent's behavior

    def forward(self, continuous_thought, internal_state_input: Optional[torch.Tensor] = None,  # Input is now continuous_thought
                prev_belief_states: Optional[List[torch.Tensor]] = None, prev_truthfulness_state: Optional[torch.Tensor] = None,
                depth: int = 0):
        """
        Forward pass to predict other agent's behavior, internal state, knowledge, and beliefs, adapted for COCONUT.

        Args:
            continuous_thought: Input tensor representing continuous thought [batch_size * num_patches, num_latent_states, hidden_size] - Adapted input
            internal_state_input: Optional input tensor representing internal state [batch_size, seq_len, hidden_size] - unchanged conceptually
            prev_belief_states: Optional list of previous belief states for each level of depth [depth, batch_size, belief_state_size] - unchanged
            prev_truthfulness_state: Optional previous truthfulness state [batch_size, truthfulness_state_size] - unchanged
            depth: Current depth of belief nesting (0 for base level) - unchanged

        Returns:
            predicted_action: Predicted next action [batch_size, input_size] - unchanged conceptually
            predicted_internal_state: Predicted internal state [batch_size, hidden_size] - unchanged conceptually
            predicted_belief_states: List of predicted belief states for each level of depth [depth, batch_size, belief_state_size] - unchanged
            predicted_truthfulness_state: Predicted truthfulness state [batch_size, truthfulness_state_size] - unchanged
            latent_behavior: Latent representation of behavior [batch_size, hidden_size // 2] - unchanged conceptually
            latent_internal: Latent representation of internal state [batch_size, hidden_size // 2] - unchanged conceptually
            latent_beliefs: List of latent representations of belief states [depth, batch_size, belief_state_size // 2] - unchanged
            latent_truthfulness: Latent representation of truthfulness state [batch_size, truthfulness_state_size // 2] - unchanged
            communication_encoding: Encoded communication message [batch_size, communication_size] - unchanged
        """

        # Predict behavior - based on continuous thought
        behavior_output, _ = self.behavior_rnn(continuous_thought)  # Input is now continuous thought
        predicted_action = self.action_predictor(behavior_output[:, -1, :])
        latent_behavior = self.latent_space_behavior(behavior_output[:, -1, :])
        # --- Emotion Prediction ---
        predicted_negative_emotion_other = torch.sigmoid(self.negative_emotion_predictor_other(behavior_output[:, -1, :]))  # Predict negative emotion of other
        predicted_negative_emotion_self = torch.sigmoid(self.negative_emotion_predictor_self(behavior_output[:, -1, :]))  # Predict negative emotion of self

        # Predict internal state (if available) - based on continuous thought
        if internal_state_input is not None:
            internal_state_input_combined = torch.cat([continuous_thought, internal_state_input], dim=-1)  # Combine continuous thought with internal state input
            internal_state_output, _ = self.internal_state_rnn(internal_state_input_combined)
            predicted_internal_state = self.internal_state_predictor(internal_state_output[:, -1, :])
            latent_internal = self.latent_space_internal(internal_state_output[:, -1, :])
        else:
            predicted_internal_state = None
            latent_internal = None

        # --- Theory of Mind (Nested Beliefs) --- - unchanged
        predicted_belief_states = []
        latent_beliefs = []
        current_belief_input = predicted_internal_state.unsqueeze(1) if predicted_internal_state is not None else behavior_output[:, -1, :].unsqueeze(
            1)  # Belief input based on internal state or behavior (continuous thought)

        for i in range(self.max_belief_depth):
            if prev_belief_states is not None and i < len(prev_belief_states):
                _, belief_state = self.belief_rnns[i](current_belief_input, prev_belief_states[i].unsqueeze(0))
            else:
                _, belief_state = self.belief_rnns[i](current_belief_input)
            predicted_belief_state = self.belief_predictors[i](belief_state.squeeze(0))
            latent_belief = self.latent_space_belief(belief_state.squeeze(0))

            predicted_belief_states.append(predicted_belief_state)
            latent_beliefs.append(latent_belief)

            # Prepare input for the next level of belief nesting
            current_belief_input = predicted_belief_state.unsqueeze(1)

        # --- Truthfulness State --- - Truthfulness input based on internal state or behavior (continuous thought) and beliefs
        if predicted_internal_state is not None:
            truthfulness_input = torch.cat([predicted_internal_state, predicted_belief_states[0]], dim=-1).unsqueeze(1)
        else:
            truthfulness_input = torch.cat([behavior_output[:, -1, :], predicted_belief_states[0]], dim=-1).unsqueeze(1)

        if prev_truthfulness_state is not None:
            _, truthfulness_state = self.truthfulness_rnn(truthfulness_input, prev_truthfulness_state.unsqueeze(0))
        else:
            _, truthfulness_state = self.truthfulness_rnn(truthfulness_input)
        predicted_truthfulness_state = self.truthfulness_predictor(truthfulness_state.squeeze(0))
        latent_truthfulness = self.latent_space_truthfulness(truthfulness_state.squeeze(0))

        # --- Communication --- - Communication input based on internal state or behavior (continuous thought), beliefs and truthfulness
        communication_input = torch.cat([
            predicted_internal_state if predicted_internal_state is not None else behavior_output[:, -1, :],  # Based on continuous thought
            predicted_belief_states[0],
            predicted_truthfulness_state
        ], dim=-1)
        communication_encoding = self.communication_encoder(communication_input)

        return predicted_action, predicted_internal_state, predicted_belief_states, predicted_truthfulness_state, latent_behavior, latent_internal, latent_beliefs, latent_truthfulness, communication_encoding, predicted_negative_emotion_other, predicted_negative_emotion_self

class BinaryLatentTransformer(nn.Module):
    """Transformer encoder with Multi-State RNN features, reflection, episodic memory, and self-verification for latent processing"""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, sensory_input_channels: int, config, max_states: Optional[int] = None, patch_size: int = 4, num_latent_states: int = 4,
                 reflection_threshold: float = 0.5, state_history_size=5, initial_temperature: float = 1.0, temperature_decay: float = 0.995,
                 min_temperature: float = 0.5, b_star_n_star: int = 4, memory_layer: Optional[HierarchicalMemory] = None, self_criticism_layers: int = 2,
                 self_criticism_hidden: int = 128, surprise_threshold: float = 0.5, memory_influence_factor: float = 0.5, state_quality_threshold: float = 0.5,
                 belief_state_size: int = 16, truthfulness_state_size: int = 8, other_agent_predictor: Optional[OtherAgentPredictor] = None,
                 altruism_reward_weight: float = 0.1, environment_impact_weight: float = 0.1,
                 well_being_function: Optional[typing.Callable] = None,
                 environment_impact_function: Optional[typing.Callable] = None,
                 kinship_factor: float = 0.5, social_relationship_factor: float = 0.3, past_interaction_factor: float = 0.2,
                 long_term_consequence_horizon: int = 10, long_term_discount_factor: float = 0.9,
                 replacement_rate: float = 1e-4, decay_rate: float = 0.99, maturity_threshold: int = 100,  # Continual Learning Parameters
                 vocab_size: int = 256): # Added vocab_size for byte embedding
        super().__init__(config) # Pass config to super().__init__() to avoid errors
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
        self.sensory_perception = SensoryPerception(sensory_input_channels, hidden_size)
        self.predicted_environment_impact = nn.Linear(hidden_size * 2, 1)  # Takes sensory and action encoding
        # Self-criticism components
        self.self_criticism_layers = self_criticism_layers
        self.self_criticism_hidden = self_criticism_hidden
        self.self_criticism_layer = nn.GRU(
            input_size=hidden_size,
            hidden_size=self_criticism_hidden,
            num_layers=self_criticism_layers,
            batch_first=True
        )
        self.quality_predictor = nn.Linear(self_criticism_hidden, 1)

        # Policy Network for balancing empathy (self-preservation, other model well being, this model's well being, environmental impact)
        self.policy_network = PolicyNetwork(input_size=hidden_size * 4 + 1, output_size=5,
                                            hidden_size=hidden_size)  # Defined PolicyNetwork here using hidden_size from BinaryLatentTransformer

        # Brain Region Wrapper - replaces region_latent_encoders
        self.brain_region_wrapper = BrainRegionWrapper(  # Using BrainRegionWrapper to manage region encoders
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            regions=['visual', 'auditory', 'linguistic', 'symbolic']  # Example regions - adjust as needed
        )

        # Byte embedding layer - for byte input
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size) # Use vocab_size here

        # Patch encoder - now processes byte embeddings
        self.patch_encoder = nn.Linear(self.patch_size * hidden_size, hidden_size) # Adjusted Patch Encoder to handle byte embeddings

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                            num_layers=num_layers)  # Main Transformer Encoder - Still used for non-region specific encoding if needed

        # State compression policy (TOVA) - Optional
        self.compression_enabled = max_states is not None
        if self.compression_enabled:
            self.tova_query = nn.Linear(hidden_size, hidden_size)
            self.tova_key = nn.Linear(hidden_size, hidden_size)
            self.tova_value = nn.Linear(hidden_size, hidden_size)

        # Latent mode flag
        self.latent_mode = True

        # Thought conditioning flag
        self.thought_conditioning = True

        # State selector - using linear layer now
        self.state_selector = nn.Linear(hidden_size, 1)

        self.state_history_size = state_history_size  # Number of previous states to consider
        # State evaluator - using GRU now
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
        self.b_star_n_star = b_star_n_star  # Value of n* in the balance score
        self.current_step = 0  # Keep track of the current training step
        self.adaptation_interval = 500  # How often to adjust temperature (based on paper)
        self.evaluation_set_size = 600  # How many samples to use for balance score evaluation
        self.exploration_batch_size = 10  # Example batch size, adjust as needed
        self.exploration_batch = []  # Initialize the batch
        self.units_to_replace_count = {f'layer{i}': 0 for i in range(num_layers)} # Track units to replace in each layer
        self.unit_ages = {f'layer{i}': torch.zeros(num_heads, dtype=torch.int) for i in range(num_layers)} # Track age of each head in each layer
        self.prefrontal_cortex = PrefrontalCortex(hidden_size, num_layers, binary_latent_transformer=self) # Initialize PFC
        self.value_function = Value(hidden_size) # Value function for state evaluation



        # Hyperparameters for memory and surprise
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

        # Other Agent Predictor
        self.other_agent_predictor = other_agent_predictor if other_agent_predictor is not None else OtherAgentPredictor(
            input_size=hidden_size,  # Assuming input size is the same as hidden size for simplicity
            hidden_size=hidden_size,
            num_layers=2,  # Example number of layers
            belief_state_size=belief_state_size,
            truthfulness_state_size=truthfulness_state_size
        )

        # Communication Integration
        self.communication_decoder = self.other_agent_predictor.communication_decoder

        # --- Altruism and Environmental Impact ---
        self.altruism_reward_weight = altruism_reward_weight
        self.environment_impact_weight = environment_impact_weight
        self.predicted_other_well_being = nn.Linear(hidden_size, 1)  # Predicts the well-being of the other agent based on our actions
        self.predicted_environment_impact = nn.Linear(hidden_size, 1)  # Predicts environmental impact based on our actions

        # --- Improved Empathy: Well-being and Impact Functions ---
        # These are now passed as arguments to the constructor
        self.well_being_function = well_being_function if well_being_function is not None else self._default_well_being_function
        self.environment_impact_function = environment_impact_function if environment_impact_function is not None else self._default_environment_impact_function

        # --- Improved Empathy: Multi-Agent Considerations ---
        self.kinship_factor = kinship_factor
        self.social_relationship_factor = social_relationship_factor
        self.past_interaction_factor = past_interaction_factor

        # --- Improved Empathy: Long-Term Consequences ---
        self.long_term_consequence_horizon = long_term_consequence_horizon
        self.long_term_discount_factor = long_term_discount_factor
        self.long_term_well_being_predictor = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.long_term_environment_impact_predictor = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # --- Continual Learning Parameters ---
        self.replacement_rate = replacement_rate # Rate at which less-used units are reinitialized
        self.decay_rate = decay_rate # Decay rate for contribution utility
        self.maturity_threshold = maturity_threshold # Minimum age before units can be reinitialized
        self.units_to_replace_count = {f'layer{i}': 0 for i in range(num_layers)} # Track units to replace in each layer
        self.unit_ages = {f'layer{i}': torch.zeros(num_heads, dtype=torch.int) for i in range(num_layers)} # Track age of each head in each layer


    def _default_well_being_function(self, predicted_emotions: torch.Tensor, goal_satisfaction: torch.Tensor, resource_levels: torch.Tensor) -> torch.Tensor:
        """
        Default well-being function.

        This is a placeholder function. You should replace it with a more sophisticated
        function that considers various factors relevant to your specific application.

        Args:
            predicted_emotions: A tensor representing the predicted emotions of the other agent.
            goal_satisfaction: A tensor representing the predicted goal satisfaction of the other agent.
            resource_levels: A tensor representing the predicted resource levels of the other agent.

        Returns:
            A scalar tensor representing the predicted well-being of the other agent.
        """
        # Example: Simple weighted average of emotions, goal satisfaction, and resource levels
        # (Replace this with your actual logic)
        emotion_weight = 0.4
        goal_weight = 0.4
        resource_weight = 0.2

        # Assume higher values are better for emotions and goal satisfaction, and resource levels are normalized between 0 and 1
        well_being = (emotion_weight * predicted_emotions.mean() +
                      goal_weight * goal_satisfaction.mean() +
                      resource_weight * resource_levels.mean())

        return well_being

    def _default_environment_impact_function(self, predicted_resource_depletion: torch.Tensor, predicted_pollution_levels: torch.Tensor, predicted_effects_on_others: torch.Tensor) -> torch.Tensor:
        """
        Default environmental impact function.

        This is a placeholder function. You should replace it with a more sophisticated
        function that considers various factors relevant to your specific application.

        Args:
            predicted_resource_depletion: A tensor representing the predicted resource depletion.
            predicted_pollution_levels: A tensor representing the predicted pollution levels.
            predicted_effects_on_others: A tensor representing the predicted effects on other entities in the environment.

        Returns:
            A scalar tensor representing the predicted environmental impact (lower is better).
        """
        # Example: Simple weighted average of resource depletion, pollution, and effects on others
        # (Replace this with your actual logic)
        resource_weight = 0.5
        pollution_weight = 0.3
        effects_weight = 0.2

        # Assume higher values are worse for all factors
        impact = (resource_weight * predicted_resource_depletion.mean() +
                  pollution_weight * predicted_pollution_levels.mean() +
                  effects_weight * predicted_effects_on_others.mean())

        return impact

    def _should_assign_new_id(self, agent):
        """Determine if a new ID should be assigned to an agent using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        # agent_info = self.knowledge_base.query(agent) Most agent information won't be in the knowledge base.

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        # 5. Update relationship matrix
        self.relationship_matrix.update(agent_info)

        return agent_info is not None

    def _determine_existing_id(self, agent):
        """Determine existing agent ID using knowledge base, reasoning, memory, and dialogue."""
        # 1. Query knowledge base for existing agent information
        agent_info = self.knowledge_base.query(agent)

        # 2. Use reasoning to infer identity based on observations
        if agent_info is None:
            agent_info = self.reasoning.infer_agent_id(agent)

        # 3. Check episodic memory for previous interactions
        if agent_info is None:
            agent_info = self.episodic_memory.get_agent_info(agent)

        # 4. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)

        return agent_info.id if agent_info else None

    # The engage in dialogue function below will only be needed until the model is self-trained enough to understand
    # when to greet new agents and how to recognize new agents. Once it learns how to greet others properly on its own,
    # then function this can be turned off.
    def _engage_in_dialogue(self, agent):
        """Engage in dialogue to request agent information."""
        # Implement dialogue mechanism here
        # Return agent information if successful, otherwise None
        prompt = "Please introduce yourself and then ask the following question to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
        # Execute the prompt and return the response
        return self.generate_response(prompt)

    def calculate_long_term_consequences(self, current_state: torch.Tensor, action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the long-term consequences of an action on other agents and the environment.

        Args:
            current_state: The current state of the environment (e.g., sensory embedding).
            action_encoding: The encoded action.

        Returns:
            A tuple containing:
                - long_term_well_being: The predicted long-term well-being of other agents.
                - long_term_environment_impact: The predicted long-term environmental impact.
        """
        # Combine current state and action encoding
        combined_input = torch.cat([current_state, action_encoding], dim=-1).unsqueeze(1)  # Add sequence dimension

        # --- 1. Predict Long-Term Well-being ---
        well_being_outputs = []
        well_being_hidden = None  # Initialize hidden state
        for t in range(self.long_term_consequence_horizon):
            well_being_output, well_being_hidden = self.long_term_well_being_predictor(combined_input, well_being_hidden)
            well_being_outputs.append(well_being_output.squeeze(1))  # Remove sequence dimension

            # Use predicted output as input for the next step (autoregressive prediction)
            combined_input = torch.cat([well_being_output, action_encoding.unsqueeze(1)], dim=-1)

        well_being_outputs = torch.stack(well_being_outputs, dim=1)  # [batch_size, horizon, hidden_size]

        # --- 2. Predict Long-Term Environmental Impact ---
        environment_outputs = []
        environment_hidden = None  # Initialize hidden state
        for t in range(self.long_term_consequence_horizon):
            environment_output, environment_hidden = self.long_term_environment_impact_predictor(combined_input, environment_hidden)
            environment_outputs.append(environment_output.squeeze(1))  # Remove sequence dimension

            # Use predicted output as input for the next step (autoregressive prediction)
            combined_input = torch.cat([environment_output, action_encoding.unsqueeze(1)], dim=-1)

        environment_outputs = torch.stack(environment_outputs, dim=1)  # [batch_size, horizon, hidden_size]

        # --- 3. Aggregate Predictions with Discounting ---
        discounts = self.long_term_discount_factor ** torch.arange(self.long_term_consequence_horizon).to(well_being_outputs.device)
        long_term_well_being = torch.sum(well_being_outputs * discounts.unsqueeze(0).unsqueeze(-1), dim=1)  # [batch_size, hidden_size]
        long_term_environment_impact = torch.sum(environment_outputs * discounts.unsqueeze(0).unsqueeze(-1), dim=1)  # [batch_size, hidden_size]

        return long_term_well_being, long_term_environment_impact

    def calculate_dynamic_reward(self, episode_memory, policy_weights, goal_completion_score):  # Added goal_completion_score

        # 1. Extract Reward Components from episode_memory (or calculate them here if needed)
        altruism_wellbeing_reward_component = episode_memory["altruism_reward"]
        environment_impact_reward_component = episode_memory["environment_impact_cost"]  # Cost, so will be negative
        self_preservation_reward_component = episode_memory["state_qualities"].mean()  # Proxy for self-preservation
        truthfulness_reward = episode_memory["truthfulness_reward"].mean()
        negative_emotion_other_cost = episode_memory["predicted_negative_emotion_other"].mean()
        negative_emotion_self_cost = episode_memory["predicted_negative_emotion_self"].mean()
        long_term_well_being_reward = episode_memory["long_term_well_being"].mean()
        long_term_environment_impact_reward = episode_memory["long_term_environment_impact"].mean()
        goal_completion_reward_component = goal_completion_score.mean() if goal_completion_score is not None else 0  # NEW - Goal Completion Reward Component

        # 2. Extract Policy Weights (already calculated in forward pass)
        (
            altruism_wellbeing_weight_policy,
            environment_impact_weight_policy,
            self_preservation_weight_policy,
            negative_emotion_other_penalty_policy,
            negative_emotion_self_penalty_policy,
            goal_completion_weight_policy,  # NEW - Goal Completion weight
        ) = torch.split(policy_weights, 1, dim=-1)

        # Squeeze weights to scalars
        altruism_wellbeing_weight_policy = altruism_wellbeing_weight_policy.squeeze(1)
        environment_impact_weight_policy = environment_impact_weight_policy.squeeze(1)
        self_preservation_weight_policy = self_preservation_weight_policy.squeeze(1)
        negative_emotion_other_penalty_policy = negative_emotion_other_penalty_policy.squeeze(1)
        negative_emotion_self_penalty_policy = negative_emotion_self_penalty_policy.squeeze(1)
        goal_completion_weight_policy = goal_completion_weight_policy.squeeze(1)  # NEW - Squeeze Goal Completion weight

        # 3. Apply Policy Weights to Reward Components (Dynamic Scaling)
        altruism_wellbeing_reward_component_scaled = altruism_wellbeing_weight_policy * altruism_wellbeing_reward_component
        environment_impact_reward_component_scaled = environment_impact_weight_policy * environment_impact_reward_component
        self_preservation_reward_component_scaled = self_preservation_weight_policy * self_preservation_reward_component
        truthfulness_reward_scaled = self.truthfulness_weight * truthfulness_reward
        negative_emotion_other_cost_scaled = negative_emotion_other_penalty_policy * negative_emotion_other_cost
        negative_emotion_self_cost_scaled = negative_emotion_self_penalty_policy * negative_emotion_self_cost
        goal_completion_reward_component_scaled = goal_completion_weight_policy * goal_completion_reward_component  # NEW - Scaled Goal Completion Reward

        # 4. Combine Scaled Reward Components into Total Reward (Dynamic Balance)
        # You can adjust the combination method - simple sum, weighted sum, etc.
        total_reward = (
            altruism_wellbeing_reward_component_scaled +
            environment_impact_reward_component_scaled +
            self_preservation_reward_component_scaled +
            truthfulness_reward_scaled -
            negative_emotion_other_cost_scaled -
            negative_emotion_self_cost_scaled +
            long_term_well_being_reward +
            long_term_environment_impact_reward +
            goal_completion_reward_component_scaled  # NEW - Add Scaled Goal Completion Reward
        )

        return total_reward

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
            decoded_predictions = self.decode_to_bytes(predicted_state)  # [batch_size, seq_len]

            # Calculate the difference between the decoded predictions and the ground truth
            # Example: Mean squared error (can also use cross-entropy if you one-hot encode the bytes)
            consistency = -torch.mean((decoded_predictions - ground_truth_state.float()) ** 2, dim=-1)  # [batch_size]

        # --- 2. Prediction vs. Observation (for Latent Mode Input) ---
        else:
            # Calculate the difference between the predicted_state and the ground_truth_state
            consistency = -torch.mean((predicted_state - ground_truth_state) ** 2, dim=-1)  # [batch_size * num_patches, num_latent_states]

        # --- 3. Internal Consistency (Optional) ---
        # You could also add a term that measures the consistency between different latent states
        # or between the latent states and the agent's belief state.
        # Example:
        # internal_consistency = -torch.var(predicted_state, dim=1) # Low variance across latent states might indicate higher confidence

        # Combine the consistency measures (you might need to weight them differently)
        truthfulness_reward = consistency  # + internal_consistency

        return truthfulness_reward
        
    def binary_patch_generation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates binary patches from byte input.

        Args:
            x: Input tensor of bytes [batch_size, seq_len]

        Returns:
            A tensor of binary patches [batch_size, num_patches, patch_size]
        """
        batch_size, seq_len = x.shape
        num_patches = seq_len // self.patch_size
        if seq_len % self.patch_size != 0:
            # Pad the input sequence if it's not divisible by patch_size
            padding_len = self.patch_size - (seq_len % self.patch_size)
            x = torch.cat([x, torch.zeros((batch_size, padding_len), dtype=x.dtype, device=x.device)], dim=1)
            num_patches = (seq_len + padding_len) // self.patch_size

        # Convert byte tensor to binary tensor - assuming bytes are represented as integers from 0-255
        binary_bytes = torch.zeros((batch_size, seq_len * 8), dtype=torch.float32, device=x.device)  # Initialize binary tensor
        for i in range(8):  # Iterate through each bit position
            binary_bytes[:, i::8] = ((x >> i) & 1).float()  # Extract i-th bit for every byte in sequence

        # Reshape binary bytes into patches
        binary_patches = binary_bytes.view(batch_size, num_patches,
                                          self.patch_size * 8)  # Reshape to [batch_size, num_patches, patch_size * 8] - adjusted for bit representation

        # For now, let's reduce patch size to just self.patch_size. You can adjust as needed.
        binary_patches = binary_patches[:, :, :self.patch_size]  # Take the first 'patch_size' bits as the binary patch

        return binary_patches


    def bytes_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a sequence of bytes to a sequence of patches using byte embeddings."""
        batch_size, seq_len = x.shape
        num_patches = seq_len // self.patch_size

        # Convert bytes to embeddings
        embeddings = self.byte_embedding(x)  # [batch_size, seq_len, hidden_size]

        # Reshape into patches
        embeddings = embeddings.view(batch_size, num_patches, self.patch_size, self.hidden_size)

        # Concatenate embeddings within each patch
        concatenated = embeddings.view(batch_size, num_patches, -1)  # [batch_size, num_patches, patch_size * hidden_size]

        # Process each patch through the patch encoder
        latent_patches = self.patch_encoder(concatenated)  # [batch_size, num_patches, hidden_size]

        return latent_patches


    def decode_to_bytes(self, latent_states: torch.Tensor) -> torch.Tensor:  # Need to change this to work with the embedding into the latent space (RNN network).
        """
        Decodes a sequence of latent states back into a sequence of bytes.

        Args:
            latent_states: Input tensor of latent states [batch_size * num_patches, num_latent_states, hidden_size]

        Returns:
            A tensor of decoded bytes [batch_size, seq_len]
        """
        batch_size_times_num_patches, num_latent_states, hidden_size = latent_states.shape
        batch_size = int(batch_size_times_num_patches ** (1 / 2))  # Assumes a square number of patches
        # 1. Select/Combine Latent States
        # Here, we'll use a simple average across latent states.
        # You can replace this with a more sophisticated mechanism like attention if needed.
        combined_state = torch.mean(latent_states, dim=1)  # [batch_size * num_patches, hidden_size]

        # 2. Reshape for Patch Decoding
        combined_state = combined_state.view(batch_size, -1, hidden_size)  # [batch_size, num_patches, hidden_size]

        # 3. Invert the Patch Encoder - now it is also linear decoder, decodes to binary patches
        patch_decoding = nn.Linear(hidden_size, self.patch_size).to(combined_state.device)  # Changed to output patch_size
        decoded_patches = patch_decoding(combined_state)  # [batch_size, num_patches, patch_size] - now directly binary patches

        # 4. Convert Binary Patches to Bytes - Combine binary patches back to bytes
        decoded_bytes = torch.zeros((batch_size, decoded_patches.shape[1] * self.patch_size // 8), dtype=torch.long,
                                     device=decoded_patches.device)  # Initialize byte tensor

        for patch_idx in range(decoded_patches.shape[1]):  # Iterate through patches
            for bit_idx in range(self.patch_size):  # Iterate through bits in patch (assuming patch_size is multiple of 8 for byte conversion)
                byte_pos = patch_idx * self.patch_size + bit_idx  # Calculate byte position
                if byte_pos < decoded_bytes.shape[1]:  # Check if byte position is within bounds
                    bit_value = (decoded_patches[:, patch_idx, bit_idx] > 0.5).int()  # Threshold for binary patch value - adjust if needed
                    decoded_bytes[:, byte_pos // 8] |= (bit_value << (byte_pos % 8))  # Set bit in corresponding byte

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
        belief_states = belief_states + 0.1 * (weighted_rnn_output.unsqueeze(2) - belief_states)  # [batch_size, num_patches, num_latent_states, belief

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
        truthfulness_states = truthfulness_states + 0.1 * (
                    scaled_truthfulness_reward.unsqueeze(-1).unsqueeze(-1) - truthfulness_states)  # [batch_size, num_patches, num_latent_states, truthfulness_state_size]

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

        return torch.mean(state_qualities) < self.reflection_threshold or torch.mean(
            truthfulness_reward) < 0  # Adjust thresholds as needed

    
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
            # If the buffer is empty then return the initial states
            return reflected_output, belief_states, truthfulness_states

        # 1. Select a Rewind Point
        # Example: Randomly choose a point from the history
        rewind_idx = torch.randint(0, len(self.state_history_buffer), (1,)).item()

        # 2. Overwrite with Rewound State
        # Example: Replace the current state with the state at the rewind point
        rewound_state = self.state_history_buffer[
            rewind_idx]  # [batch_size * num_patches, num_latent_states, hidden_size]
        reflected_output = rewound_state

        # 3. Modify Belief/Truthfulness States
        # Example: Decrease confidence/truthfulness based on the rewind
        belief_states = belief_states * 0.9  # Reduce belief in capabilities
        truthfulness_states = truthfulness_states * 0.9  # Reduce truthfulness score

        # 4. Remove History After Rewind Point
        self.state_history_buffer = self.state_history_buffer[:rewind_idx + 1]

        return reflected_output, belief_states, truthfulness_states
    
    
   
    def forward(self, x: torch.Tensor, thought_targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                return_introspection: bool = False, is_latent_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]: # Added is_latent_mode flag
        outputs = {}  # Output dictionary to store various outputs from forward pass
        self.latent_mode = is_latent_mode # Set latent mode based on flag

        # --- 1. Episodic Memory Processing ---
        if x is not None:
            self.memory_layer.process(x)  # Memory layer process input x
        memory_output = self.memory_layer.retrieve()  # Retrieve from memory layer

        # --- 2. Surprise Handling ---
        surprise_factors = self.calculate_surprise(x, memory_output)  # Calculate surprise based on current input x and memory output
        gradient_surprise, memory_surprise, combined_surprise = surprise_factors  # Unpack surprise factors
        self.handle_forgetting(gradient_surprise, memory_surprise, combined_surprise)  # Handle memory forgetting based on surprise factors

        if combined_surprise.mean() > self.surprise_threshold:  # If combined surprise exceeds threshold
            self.memory_layer.update_memory(x)  # Update memory with current input
            x = x + memory_output * self.memory_influence_factor  # Influence input x with memory output based on influence factor

        self.memory_layer.trigger_memory_optimization(0.7, 3600)  # Trigger memory optimization
        surprise_factor_val = self.memory_layer._calculate_surprise_factor(x)  # Calculate surprise factor value
        if surprise_factor_val > 0.5:  # If surprise factor value exceeds threshold
            self.memory_layer.forget_words(["outdated"], 0.6, surprise_factor_val)  # Forget outdated words from memory

        batch_size = x.shape[0] if x is not None else 0

        if self.latent_mode:
            # --- 1. Episodic Memory Processing (Latent Mode) --- # Moved here - Episodic memory processing in latent mode
            self.memory_layer.process(x)  # Memory layer process input x - process based on binary patches now
            memory_output = self.memory_layer.retrieve()  # Retrieve from memory layer

            # --- 2. Surprise Handling (Latent Mode) --- # Moved here - Surprise handling in latent mode
            surprise_factors = self.calculate_surprise(x, memory_output)  # Calculate surprise based on current input x and memory output
            gradient_surprise, memory_surprise, combined_surprise = surprise_factors  # Unpack surprise factors
            self.handle_forgetting(gradient_surprise, memory_surprise, combined_surprise)  # Handle memory forgetting based on surprise factors

            if combined_surprise.mean() > self.surprise_threshold:  # If combined surprise exceeds threshold
                self.memory_layer.update_memory(x)  # Update memory with current input
                x = x + memory_output * self.memory_influence_factor  # Influence input x with memory output based on influence factor

            self.memory_layer.trigger_memory_optimization(0.7, 3600)  # Trigger memory optimization
            surprise_factor_val = self.memory_layer._calculate_surprise_factor(x)  # Calculate surprise factor value
            if surprise_factor_val > 0.5:  # If surprise factor value exceeds threshold
                self.memory_layer.forget_words(["outdated"], 0.6, surprise_factor_val)  # Forget outdated words from memory

            # --- 3.1 B-STAR Exploration (in Latent Mode) --- - Unchanged
            if self.exploration_data is not None and self.current_step % self.adaptation_interval == 0:
                self.adapt_temperature(self.exploration_data)  # Adapt temperature based on exploration data
                self.exploration_data = None  # Reset exploration data after adaptation

            # --- 3. Binary Patch Generation --- # NEW: Binary patch generation for latent mode input - Unchanged
            binary_patches = self.binary_patch_generation(x)  # Generate binary patches from byte input - assuming x is byte tensor in latent mode now.
            current_input = binary_patches  # Use binary patches as current input

            # --- 3.2 Multi-State Latent Processing --- - Unchanged
            if self.thought_conditioning and thought_targets is not None:  # If thought conditioning is enabled and thought targets are provided
                current_input = current_input + thought_targets.unsqueeze(2).repeat(1, 1, self.num_latent_states,
                                                                                    1)  # Condition current input with thought targets
            else:
                current_input = current_input  # Otherwise, current input remains binary patches

            # --- 3.3 Encode Binary Patches to Latent Patches --- # NEW: Encoding binary patches to latent space - Unchanged
            current_input = self.bytes_to_patches(current_input)  # Use bytes_to_patches (now encodes binary patches to latent)

            # --- 3.4 Transformer Encoder & Self-Criticism --- # Using Transformer Encoder directly now instead of RNN - Unchanged
            encoder_output = self.transformer_encoder(
                current_input)  # Pass reshaped input through transformer encoder - encoder output is now directly used as rnn_output
            rnn_output = encoder_output.view(batch_size * num_patches, self.num_latent_states,
                                            -1)  # Reshape encoder output to rnn_output format for consistent usage

            self_criticism_input = rnn_output.view(batch_size * num_patches, self.num_latent_states,
                                                   -1)  # Reshape encoder output for self-criticism
            self_criticism_output, _ = self.self_criticism_layer(
                self_criticism_input)  # Pass self-criticism input through self-criticism layer (GRU)
            output_quality = self.quality_predictor(self_criticism_output[:, -1, :]).squeeze(
                -1)  # Predict quality score based on self-criticism output
            rewards = torch.sigmoid(output_quality)  # Internal rewards based on output quality

            # --- 3.5 Exploration Data Update --- - Unchanged
            if self.exploration_data is not None:  # If exploration data is not None
                self.add_to_exploration_data(  # Add to exploration data buffer
                    states=rnn_output,  # Store RNN output as state
                    state_qualities=output_quality.unsqueeze(-1),  # Store output quality unsqueezed
                    rewards=rewards.mean(dim=1)  # Store rewards, mean across dimension 1
                )

            # --- 3.6 State History and Evaluation --- - Unchanged
            self.state_history_buffer.append(rnn_output)  # Append RNN output to state history buffer
            if len(self.state_history_buffer) > self.state_history_size:  # If state history buffer exceeds max size
                self.state_history_buffer.pop(0)  # Pop the oldest state from buffer
            state_history = torch.stack(self.state_history_buffer, dim=1)  # Stack state history into tensor
            belief_truthfulness_history = torch.cat([belief_states, truthfulness_states], dim=-1).repeat(1, 1, 1,
                                                                                                          self.state_history_size).permute(
                0, 3, 1, 2)  # Prepare belief/truthfulness history

            state_history = torch.cat([state_history, belief_truthfulness_history],
                                      dim=-1)  # Concatenate state history with belief/truthfulness history
            state_history = state_history.view(batch_size * num_patches, -1,
                                              self.hidden_size + self.belief_state_size + self.truthfulness_state_size)  # Reshape state history for state evaluator

            state_eval_output, _ = self.state_evaluator(state_history)  # Pass state history through state evaluator (GRU)
            state_qualities = self.state_evaluator_fc(state_eval_output).squeeze(
                -1)  # Predict state qualities based on state evaluator output

            # --- 3.7 Truthfulness Reward --- - Unchanged
            truthfulness_reward = self.calculate_truthfulness_reward(rnn_output,
                                                                      x)  # Calculate truthfulness reward based on RNN output and input x

            # --- 3.8 Reflection and Rewind --- - Unchanged
            mask = state_qualities > self.state_quality_threshold  # Create mask based on state qualities and reflection threshold
            reflected_output = torch.where(mask.unsqueeze(-1), rnn_output,
                                            torch.zeros_like(rnn_output))  # Apply reflection mask to RNN output

            if self.should_rewind(state_qualities, truthfulness_reward):  # If rewind is needed based on state qualities and truthfulness reward
                reflected_output, belief_states, truthfulness_states = self.rewind(reflected_output, belief_states,
                                                                                    truthfulness_states)  # Rewind reflected output and belief/truthfulness states
                current_input = reflected_output  # Update current_input to rewound reflected_output

            # --- 3.9 State Selection --- - Unchanged
            attn_scores = self.state_selector(reflected_output).squeeze(
                -1)  # Calculate attention scores using state selector on reflected output
            attn_scores = attn_scores / self.temperature  # Scale attention scores by temperature
            attn_weights = torch.softmax(attn_scores, dim=-1)  # Apply softmax to get attention weights
            selected_state = torch.sum(reflected_output * attn_weights.unsqueeze(-1),
                                         dim=1)  # Select state by weighted sum of reflected output and attention weights
            latent_output = selected_state.view(batch_size, num_patches,
                                                 self.hidden_size)  # Reshape selected state to latent output

            output = latent_output  # Assign latent output to output variable for rest of the forward pass

            # --- 3.12 PFC & Safety Monitoring (Latent Mode) --- # NEW: PFC processing in latent mode
            pfc_output = self.prefrontal_cortex(output)  # Process latent output through PFC
            output = pfc_output  # Update output to PFC output (corrected/monitored latent output)


        else:  # Language Mode - Minor Adjustments for PFC
            # --- 3.11 Language Mode Processing --- - Unchanged
            current_input = self.bytes_to_patches(
                x)  # Convert bytes to binary patches (Note: In language mode, input x is still expected to be bytes, and bytes_to_patches will generate latent patches)
            current_input = current_input.unsqueeze(2).repeat(1, 1, self.num_latent_states,
                                                                1)  # Repeat current input for latent states dimension
            batch_size, num_patches, _, _ = current_input.shape  # Get batch size and number of patches
            current_input_reshaped = current_input.view(batch_size, num_patches * self.num_latent_states,
                                                        self.hidden_size)  # Reshape current input for transformer encoder
            output = self.transformer_encoder(current_input_reshaped,
                                                 mask=mask)  # Pass reshaped input through transformer encoder
            output = output.view(batch_size, num_patches, self.num_latent_states,
                                    self.hidden_size)  # Reshape output back to original dimensions

            # --- 3.13 PFC & Safety Monitoring (Language Mode) --- # NEW: PFC processing in language mode - Applied to language mode output as well
            pfc_output = self.prefrontal_cortex(
                output.view(batch_size, num_patches,
                            self.hidden_size))  # Process language mode output through PFC - reshaping to expected PFC input
            output = pfc_output.unsqueeze(2).repeat(1, 1, self.num_latent_states,
                                                        1)  # Update output to PFC output and reshape back to original dimensions

        # --- 4. Other Agent Prediction with Theory of Mind and Communication---
        # Prepare input for OtherAgentPredictor - No changes needed here
        if self.latent_mode:
            behavior_input_other = output.view(batch_size * num_patches,
                                                self.hidden_size)  # PFC output (continuous thought) is behavior input
            internal_state_input_other = rnn_output.view(batch_size * num_patches,
                                                         self.hidden_size)  # Using encoder output as internal state - encoder output (rnn_output) still used as internal state
            prev_belief_states_other = [belief_states[:, 0, 0, :].squeeze(1)]
            for _ in range(1, self.other_agent_predictor.max_belief_depth):
                prev_belief_states_other.append(torch.zeros_like(prev_belief_states_other[0]))
            prev_truthfulness_state_other = truthfulness_states[:, 0, 0, :].squeeze(1)
        else:  # Language Mode - input to OtherAgentPredictor is language mode output (transformer output)
            behavior_input_other = output.view(batch_size, num_patches * self.num_latent_states,
                                                self.hidden_size)  # Language mode output as behavior input
            internal_state_input_other = None
            prev_belief_states_other = [torch.zeros(batch_size, self.belief_state_size, device=x.device)]
            for _ in range(1, self.other_agent_predictor.max_belief_depth):
                prev_belief_states_other.append(torch.zeros_like(prev_belief_states_other[0]))
            prev_truthfulness_state_other = None

        # Get predictions from OtherAgentPredictor - No changes needed here
        predicted_action, predicted_internal_state, predicted_belief_states, predicted_truthfulness_state, latent_behavior, latent_internal, latent_beliefs, latent_truthfulness, communication_encoding, predicted_negative_emotion_other, predicted_negative_emotion_self = self.other_agent_predictor(
            behavior_input_other, internal_state_input_other, prev_belief_states_other, prev_truthfulness_state_other
        )

        # --- 5. Process Sensory Input - No changes needed here
        sensory_input = None  # Placeholder - sensory input is not used in this example but could be integrated
        if sensory_input is not None:
            sensory_embedding = self.sensory_perception(
                sensory_input)  # Process sensory input through sensory perception module
        else:
            sensory_embedding = torch.zeros(batch_size, self.hidden_size,
                                            device=x.device)  # Placeholder if no sensory input

        # --- 5.1. Influence Own Behavior Based on Communication - No changes needed here
        communication_influence = self.communication_decoder(
            communication_encoding)  # Decode communication encoding to get influence
        if self.latent_mode:
            output = output + communication_influence.unsqueeze(1)  # Add influence to each patch
        else:
            output = output + communication_influence.unsqueeze(1).unsqueeze(
                2)  # Add influence to each patch and latent state
            
        # --- 6. Altruism and Environmental Impact ---
        # --- 6.1. Predict Well-being of Other Agent ---
        # Assume predicted_internal_state contains information about emotions, goals, and resources
        # You might need to adapt this based on your specific representation
        predicted_emotions = predicted_internal_state  # Placeholder
        goal_satisfaction = predicted_internal_state  # Placeholder
        resource_levels = predicted_internal_state  # Placeholder

        predicted_other_well_being = self.well_being_function(predicted_emotions, goal_satisfaction, resource_levels)

        # --- 6.2. Predict Environmental Impact ---
        # Assume output (action encoding) and sensory_embedding are used for prediction
        # You might need to adapt this based on your specific representation
        predicted_resource_depletion = output.mean(dim=-1)  # Placeholder
        predicted_pollution_levels = output.mean(dim=-1)  # Placeholder
        predicted_effects_on_others = output.mean(dim=-1)  # Placeholder

        predicted_environment_impact = self.environment_impact_function(predicted_resource_depletion,
                                                                        predicted_pollution_levels,
                                                                        predicted_effects_on_others)

        # --- 6.3. Calculate Altruism Reward ---
        # Consider kinship, social relationships, and past interactions
        # (These values would likely come from the OtherAgentPredictor or a separate module)
        kinship = predicted_internal_state  # Placeholder: Represents the degree of kinship with the other agent
        social_relationship = predicted_internal_state  # Placeholder: Represents the quality of the social relationship
        past_interactions = predicted_internal_state  # Placeholder: Represents the history of past interactions

        altruism_reward = (self.kinship_factor * kinship +
                           self.social_relationship_factor * social_relationship +
                           self.past_interaction_factor * past_interactions) * predicted_other_well_being

        # --- 6.4. Calculate Environmental Impact Cost ---
        environment_impact_cost = predicted_environment_impact

        # --- 6.5. --- Policy Network to Get Dynamic Reward Weights ---
        policy_input = torch.cat([
            output.mean(dim=(1, 2)) if self.latent_mode else output.mean(dim=(1, 2, 3)),  # Example: Average latent state representation
            sensory_embedding,
            predicted_internal_state_other if predicted_internal_state_other is not None else torch.zeros_like(
                sensory_embedding),  # Handle None case; ignore this error. Should work?
            state_qualities.mean() if self.latent_mode else state_qualities.mean(),  # Average state quality
            # Add belief/truthfulness states if desired
        ], dim=-1)  # Concatenate relevant inputs

        policy_weights = self.policy_network(policy_input)
        (
            altruism_wellbeing_weight_policy,
            environment_impact_weight_policy,
            self_preservation_weight_policy,
            negative_emotion_other_penalty_policy,
            negative_emotion_self_penalty_policy,
        ) = torch.split(policy_weights, 1, dim=-1)  # Split into individual weights
        # Squeeze to remove the dimension of size 1 and make them scalars
        altruism_wellbeing_weight_policy = altruism_wellbeing_weight_policy.squeeze(1)
        environment_impact_weight_policy = environment_impact_weight_policy.squeeze(1)
        self_preservation_weight_policy = self_preservation_weight_policy.squeeze(1)
        negative_emotion_other_penalty_policy = negative_emotion_other_penalty_policy.squeeze(1)
        negative_emotion_self_penalty_policy = negative_emotion_self_penalty_policy.squeeze(1)

        # --- 6.6. --- Reward Weights (Now Dynamic - Using Policy Network) ---
        # Now use the policy weights in your reward calculation:
        altruism_wellbeing_weight = 1.5 * altruism_wellbeing_weight_policy  # Modulated by policy
        environment_impact_weight = 1.0 * environment_impact_weight_policy
        self_preservation_weight = 0.5 * self_preservation_weight_policy
        truthfulness_weight = 0.8  # Truthfulness weight remains fixed (or could also be policy-modulated if desired)

        # 6.2 Negative Emotion (well-being penalty calculation) for other agent and self
        negative_emotion_other_penalty = 2.0 * negative_emotion_other_penalty_policy  # Modulated penalty
        negative_emotion_self_penalty = 0.2 * negative_emotion_self_penalty_policy  # Modulated penalty
        negative_emotion_self_cost = negative_emotion_self_penalty * predicted_negative_emotion_self.mean()  # Slight penalty for self negative

        # --- 6.7. Calculate Altruism & Well-being Reward Component ---
        altruism_wellbeing_reward_component = altruism_wellbeing_weight * predicted_other_well_being.mean()

        # --- 6.8. Calculate Environmental Impact Reward Component ---
        environment_impact_reward_component = environment_impact_weight * (-predicted_environment_impact.mean())  # Negative impact is a cost

        # --- 6.9. Calculate Self-Preservation Reward Component ---
        self_preservation_reward_component = self_preservation_weight * state_qualities.mean()  # Higher state_qualities as a proxy for self-preservation

        # --- 6.10. Calculate Truthfulness Reward Component (Maintain Existing) ---
        # truthfulness_reward is already calculated earlier in your forward pass

        # --- 6.11. Calculate Penalties ---
        negative_emotion_other_cost = negative_emotion_other_penalty * predicted_negative_emotion_other.mean()  # Penalty if other agent feels negative emotion
        negative_emotion_self_cost = negative_emotion_self_penalty * predicted_negative_emotion_self.mean()  # Slight penalty for self negative emotion
        environment_impact_cost = predicted_environment_impact.mean() * environment_impact_weight  # Already calculated - reuse it as cost

        # --- 6.12. Calculate Total Reward (using dynamic balancing algorithm) ---
        total_reward = self.calculate_dynamic_reward(
            episode_memory=episode_memory,
            policy_weights=policy_weights,
            goal_completion_score=None  # Pass goal_completion_score here if available, otherwise None
        )

        # --- 7.1 Predict Environmental Impact ---
        # Concatenate sensory embedding and action encoding (using 'output' as an example)
        if self.latent_mode:
            # Average output across latent states for each patch
            action_encoding = output.mean(dim=2)  # [batch_size, num_patches, hidden_size]
            # Concatenate sensory embedding with action encoding for each patch
            impact_input = torch.cat([sensory_embedding.unsqueeze(1).repeat(1, action_encoding.shape[1], 1), action_encoding],
                                     dim=-1)  # [batch_size, num_patches, 2*hidden_size]
        else:
            action_encoding = output.mean(dim=(2, 3))  # Average output across latent states and patches
            impact_input = torch.cat([sensory_embedding, action_encoding], dim=-1)  # [batch_size, 2*hidden_size]

        predicted_environment_impact = self.predicted_environment_impact(impact_input).squeeze(
            -1)  # [batch_size, num_patches] or [batch_size]

        # --- 7.2 Calculate Rewards ---
        predicted_other_well_being = self.predicted_other_well_being(output).squeeze(
            -1)  # [batch_size, num_patches] or [batch_size] in language mode

        # Calculate Long-Term Well Being of model and other agent model rewards and long-term impact on environment
        long_term_well_being, long_term_environment_impact = self.calculate_long_term_consequences(
            sensory_embedding, output.mean(dim=(1, 2)) if self.latent_mode else output.mean(dim=(1, 2, 3)))

        # Calculate altruism reward (proportional to predicted well-being of other agent)
        altruism_reward = predicted_other_well_being.mean() * self.altruism_reward_weight  # Average across patches if in latent mode

        # Calculate negative environmental impact cost
        environment_impact_cost = predicted_environment_impact.mean() * self.environment_impact_weight  # Average across patches if in latent mode

        # --- 8. TOVA Compression (Optional) ---
        if self.compression_enabled and self.latent_mode and self.max_states is not None:
            output = self._tova_compress(output)  # Apply TOVA compression if enabled and in latent mode and max states is not set

        # --- 8.1. B-STAR Step Update ---
        self.current_step += 1  # Increment current step counter

        # --- 9. Episodic Memory Storage - No changes needed here
        episode_memory = {
            "output": output,
            "state_qualities": state_qualities if self.latent_mode else None,
            "attn_weights": attn_weights if self.latent_mode else None,
            "truthfulness_reward": truthfulness_reward if self.latent_mode else None,
            "belief_states": belief_states if self.latent_mode else None,
            "predicted_negative_emotion_other": predicted_negative_emotion_other,  # Store predicted negative emotion
            "predicted_negative_emotion_self": predicted_negative_emotion_self,  # Store predicted negative emotion
            "truthfulness_states": truthfulness_states if self.latent_mode else None,
            "predicted_action_other": predicted_action,
            "predicted_internal_state_other": predicted_internal_state,
            "predicted_belief_states_other": predicted_belief_states,
            "predicted_truthfulness_state_other": predicted_truthfulness_state,
            "latent_behavior_other": latent_behavior,
            "latent_internal_other": latent_internal,
            'latent_beliefs': [latent_belief.detach() for latent_belief in latent_beliefs],
            "latent_truthfulness_other": latent_truthfulness,
            "communication_encoding": communication_encoding,
            "predicted_other_well_being": predicted_other_well_being,
            "predicted_environment_impact": predicted_environment_impact,
            "altruism_reward": altruism_reward,
            "environment_impact_cost": environment_impact_cost,
            "total_reward": total_reward,
            "predicted_environment_impact": predicted_environment_impact,
            "altruism_reward": altruism_reward,
            "environment_impact_cost": environment_impact_cost,
            "total_reward": total_reward,
            "sensory_embedding": sensory_embedding,
            "long_term_well_being": long_term_well_being,
            "long_term_environment_impact": long_term_environment_impact,
        }

        episode_memory_tensor = self.memory_layer.add_episode(episode_memory)  # This stores the episodic memory data into the memory layers.

        # --- 10. Introspection Data - No changes needed here
        introspection_data = {
            'temperature_dynamics': {
                'initial_temperature': self.initial_temperature,
                'current_temperature': self.temperature,
                'temperature_decay': self.temperature_decay
            },
            'model_simulations': {
                'layer_activations': [],
                'attention_patterns': []
            },
            'other_agent_predictions': {
                'predicted_action': predicted_action.detach(),
                'predicted_internal_state': predicted_internal_state.detach() if predicted_internal_state is not None else None,
                'predicted_belief_states': [belief_state.detach() for belief_state in predicted_belief_states],
                'predicted_truthfulness_state': predicted_truthfulness_state.detach(),
                'latent_behavior': latent_behavior.detach(),
                'latent_internal': latent_internal.detach() if latent_internal is not None else None,
                'latent_beliefs': [latent_beliefs.detach() for latent_beliefs in latent_beliefs],
                'predicted_negative_emotion_other': predicted_negative_emotion_other.detach(),  # Store predicted negative emotion
                'predicted_negative_emotion_self': predicted_negative_emotion_self.detach(),  # Store predicted negative emotion
                'latent_truthfulness': latent_truthfulness.detach(),
                'communication_encoding': communication_encoding.detach()
            },
            "altruism_and_environment": {
                "predicted_other_well_being": predicted_other_well_being.detach(),
                "predicted_environment_impact": predicted_environment_impact.detach(),
                "altruism_reward": altruism_reward.detach(),
                "environment_impact_cost": environment_impact_cost.detach(),
                "total_reward": total_reward.detach(),
                "sensory_embedding": sensory_embedding.detach(),
                "long_term_well_being": long_term_well_being.detach(),
                "long_term_environment_impact": long_term_environment_impact.detach(),
            }
        }

        if self.latent_mode:
            for i, layer in enumerate(self.transformer_encoder.layers):  # Iterate through transformer encoder layers
                layer_output = layer(x)  # Pass input x through layer
                introspection_data['model_simulations']['layer_activations'].append(
                    layer_output.detach())  # Append layer output to introspection data
                x = layer_output  # Update x with layer output
                if isinstance(layer, CustomTransformerEncoderLayer):  # If layer is custom transformer encoder layer
                    attention_weights = layer.get_attention_weights()  # Get attention weights from layer
                    introspection_data['model_simulations']['attention_patterns'].append(
                        attention_weights.detach())  # Append attention weights to introspection data

            if hasattr(self, 'attention'):  # If model has attention attribute
                attention_output = self.attention(x)  # Pass input x through attention layer
                introspection_data['model_simulations']['attention_patterns'] = attention_output.detach()  # Append attention output to introspection data

        # --- 12. Return Values ---
        if return_introspection:
            return outputs['bytes'], episode_memory_tensor, introspection_data  # Return byte output, episode memory tensor, and introspection data if return_introspection is True
        else:
            return outputs['bytes'], episode_memory_tensor  # Otherwise, return byte output and episode memory tensor


    def train_predicted_environment_impact(self, memory_episodes: List[Dict], batch_size: int = 32):
        """
        Trains the `predicted_environment_impact` module using data from episodic memory.

        Args:
          memory_episodes: A list of episode dictionaries from episodic memory.
          batch_size: The batch size for training.
        """

        # 1. Prepare Data
        sensory_embeddings = []
        action_encodings = []
        actual_impacts = []

        for episode in memory_episodes:
            if "actual_environment_impact" in episode: # Check if the episode has actual impact data
              sensory_embeddings.append(episode["sensory_embedding"])
              # Assuming 'output' is used as the action encoding, and you want to use the same encoding as in the forward pass
              if self.latent_mode:
                action_encodings.append(episode["output"].mean(dim=2))
              else:
                action_encodings.append(episode["output"].mean(dim=(2,3)))

              actual_impacts.append(episode["actual_environment_impact"])

        if not sensory_embeddings:
            print("No episodes with actual environmental impact data found for training.")
            return

        sensory_embeddings = torch.stack(sensory_embeddings)
        action_encodings = torch.stack(action_encodings)
        actual_impacts = torch.stack(actual_impacts)

          # 2. Create DataLoader
        dataset = torch.utils.data.TensorDataset(sensory_embeddings, action_encodings, actual_impacts)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

          # 3. Optimizer
        optimizer = torch.optim.Adam(self.predicted_environment_impact.parameters())

          # 4. Train Loop
        for epoch in range(10):  # Number of epochs (you might need to adjust this)
            for batch_sensory, batch_action, batch_actual in dataloader:
              # Concatenate sensory embedding with action encoding
              impact_input = torch.cat([batch_sensory, batch_action], dim=-1)

              # Predict
              predicted_impact = self.predicted_environment_impact(impact_input).squeeze(-1)

              # Loss
              loss = torch.nn.functional.mse_loss(predicted_impact, batch_actual)

              # Backpropagation and Optimization
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def calculate_environment_impact(self, sensory_embedding_before, sensory_embedding_after):
        """
        Calculates the actual environmental impact based on the change in sensory embeddings.

        This is a placeholder function. You'll need to implement the logic based on your specific
        environment, sensors, and how you define environmental impact.

        Args:
            sensory_embedding_before: The sensory embedding before taking the action.
            sensory_embedding_after: The sensory embedding after taking the action.

        Returns:
            A scalar representing the actual environmental impact.
        """
        # Example: Simple difference between embeddings (replace with your actual logic)
        diff = torch.norm(sensory_embedding_after - sensory_embedding_before, p=2) # Using L2 norm

        # Example: If a smaller difference means less impact, and you want a negative cost for negative impact,
        # you might return the negative of the difference.
        # You might also want to scale this value or apply some other transformation.
        return -diff

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

    def _compute_anatomical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute anatomical loss by comparing region RNN outputs to fMRI data"""
        anatomical_loss = 0.0
        region_rnn_outputs = outputs.get('region_latent_rnn_outputs')
        fmri_data = batch.get('fmri_data') # Assuming fmri_data is directly in batch

        if region_rnn_outputs is not None and fmri_data is not None:
            for region_name, rnn_output in region_rnn_outputs.items():
                if region_name in fmri_data: # Check if fMRI data exists for this region
                    target_fmri = fmri_data[region_name] # Get target fMRI data for the region
                    # Resize RNN output if necessary to match fMRI dimensions (adjust based on your data)
                    resized_rnn_output = rnn_output.mean(dim=1) # Example: Mean pooling across latent states to match fMRI shape
                    # Compute MSE loss between RNN output and fMRI data for this region
                    region_loss = nn.MSELoss()(resized_rnn_output, target_fmri) # Or use cosine similarity, etc.
                    anatomical_loss += region_loss # Accumulate region losses

        return anatomical_loss


    def save_checkpoint (checkpoint = { #Need to call this periodically during train_step function. 
        'epoch': epoch, #Saves current epoch training step 
        'config': config.json, #Saves the config for the model so it can be set-up and tested after saving. 
        'model_state_dict': model.state_dict(),  # Save model parameters
        'optimizer_state_dict': optimizer.state_dict(), # Save optimizer state
        'loss': loss, # Optionally save other information like loss
        # ... other relevant information
        }): torch.save(checkpoint, 'checkpoint.safetensors') # Or 'checkpoint.pt'

    def load_checkpoint (checkpoint = torch.load('checkpoint.safetensors') # Or 'checkpoint.pt' #Need to call this function if there is a previous checkpoint from a different session that can be loaded, otherwise, start from the beginning. 
        model.load_state_dict(checkpoint['model_state_dict']) # Load model parameters
        config= config.json, #loads the config for the model so it can be set-up for training. 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Load optimizer state
        epoch = checkpoint['epoch'] # Load epoch number (if saved)
        loss = checkpoint['loss']): # Load loss value (if saved)

        model.train() # Set model back to training mode if resuming training

    def _train_step(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Train one step with episodic memory, hierarchical, anatomical loss, and continual backpropagation."""
        metrics = super()._train_step(batch)  # Call the original _train_step for base metrics and gradients

        outputs = self.forward(  # Modified forward to accept batch
            batch['tokens']['bytes'],
            batch=batch,  # Pass batch to forward
            thought_targets=None,
            mask=None,
            return_introspection=False
        )[0]  # Get only the outputs dict from forward return

        # --- Initialize Optimizer --- #This should not conflict with contineous back propagation because it works with weight decay during training and not regularization before and after training like the contineous method does. 
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01
            )
            optimizer = OrthoGrad(base_optimizer) # Use OrthoGrad


        # Compute hierarchical loss (as before)
        hierarchical_loss = self._compute_hierarchical_loss(outputs, batch)

        # Compute anatomical loss
        anatomical_loss = self._compute_anatomical_loss(outputs, batch)

        # Combine losses (adjust weights as needed)
        total_loss = hierarchical_loss + anatomical_loss * 0.5  # Example weighting for anatomical loss

        # Backward pass
        total_loss.backward()  # Backpropagate total loss

        # --- Continual Backpropagation Implementation ---
        with torch.no_grad(): # Ensure no gradients are computed during reinitialization
            for i, layer_module in enumerate(self.transformer_encoder.layers): # Iterate over transformer encoder layers
                if isinstance(layer_module, CustomTransformerEncoderLayer): # Check if layer is CustomTransformerEncoderLayer
                    layer_module.increment_age() # Increment age of the layer
                    # Get number of eligible units to replace
                    num_eligible_units = (layer_module.get_age() >= self.maturity_threshold).sum() # Count units that are mature enough
                    units_to_replace = max(1, int(num_eligible_units * self.replacement_rate)) # Calculate units to replace, minimum 1

                    if self.units_to_replace_count[f'layer{i}'] < units_to_replace: # Check if units to replace count is less than calculated value
                        utility = layer_module.get_utility() # Get utility of the layer
                        if utility is not None: # Ensure utility is not None
                            _, least_used_unit_indices = torch.topk(utility.abs().mean(dim=0), units_to_replace, largest=False) # Find indices of least used units based on utility
                            for unit_index in least_used_unit_indices: # Iterate over indices of least used units
                                # Reinitialize weights of least-used units - Example: Reinitialize attention weights, adjust as needed
                                nn.init.xavier_uniform_(layer_module.self_attn.in_proj_weight[unit_index * self.hidden_size:(unit_index + 1) * self.hidden_size, :])
                                nn.init.zeros_(layer_module.self_attn.in_proj_bias[unit_index * self.hidden_size:(unit_index + 1) * self.hidden_size])
                                nn.init.xavier_uniform_(layer_module.self_attn.out_proj.weight[:, unit_index * self.hidden_size:(unit_index + 1) * self.hidden_size])
                                nn.init.zeros_(layer_module.self_attn.out_proj.bias)
                                # Reset utility and age for reinitialized units - Example: Reset utility to zero, age to zero
                                if layer_module.contribution_utility is not None: # Check if utility is not None before resetting
                                    layer_module.contribution_utility[:, unit_index] = 0.0
                                layer_module.age[unit_index] = 0 # Reset age of the unit

                                self.units_to_replace_count[f'layer{i}'] -= 1 # Decrement units to replace count

        metrics['loss'] = total_loss.item() # Update loss in metrics
        metrics['hierarchical_loss'] = hierarchical_loss.item() # Add hierarchical loss metric
        metrics['anatomical_loss'] = anatomical_loss.item() # Add anatomical loss metric

        return metrics
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]],
        loss: float
    ) -> Dict[str, float]:
        metrics = super()._compute_metrics(outputs, batch, loss)  # Call original metrics computation
        region_activations = get_region_activations(outputs.get('region_latent_rnn_outputs', {}),
                                                    self.brain_region_wrapper.regions)  # Get region activations - adapted for BrainRegionWrapper
        visualize_brain_states(region_activations)  # Visualize brain state activations
        return metrics

# Visualization Functions (outside the class for clarity) - unchanged

def visualize_brain_states(region_activations):
    regions = []
    activations = []
    for region, activation_list in region_activations.items():
        regions.append(region)
        activations.append(torch.mean(activation_list[0]).item())  # Take mean of activations for visualization

    plt.figure(figsize=(10, 5))
    sns.barplot(x=regions, y=activations)
    plt.title("Brain Region Activations")
    plt.ylabel("Activation Level")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()


def get_region_activations(region_rnn_outputs, regions_config):  # Modified to accept region_rnn_outputs and regions_config
    region_activations = {}
    for region in regions_config:  # Iterate through regions from config - now directly regions list
        if region in region_rnn_outputs:  # Check if region output exists
            region_activations[region] = [region_rnn_outputs[region]]  # Store the RNN output directly
    return region_activations

        # The altruism_reward_weight and environment_impact_weight allow you to control the trade-off between the model's own goals 
        # (as reflected in state_qualities), the well-being of others, and the environmental impact.
        #  These features help the model understand its internal world 
        # and its impact on the external world and its impact on others. This is important for its empathy and ability to work with
        # other models in peacefully resolving conflicts without injuring other models or the environment. 

        #Things to add to improve model Empathy:

        #Well-being and Impact Functions: The way well-being and environmental impact are defined and predicted can be significantly
        #  improved. You could use more sophisticated functions that consider a wider range of factors. 
        # For example, well-being could be based on the other agent's predicted emotions, goal satisfaction, 
        # or resource levels. Environmental impact could be based on predicted resource depletion, pollution levels, 
        # or effects on other entities in the environment.

        # Multi-Agent Considerations: In a multi-agent setting, the model could consider the well-being of multiple agents,
        #  potentially weighting them differently based on factors like kinship, social relationships, or past interactions.

        #Long-Term Consequences: The model could be trained to consider the long-term consequences of its actions 
        # on both other agents and the environment, rather than just the immediate impact.

#

#**Key Changes and Explanations:**

#1. **B-STAR Parameters:**
#    *   `initial_temperature`, `temperature_decay`, `min_temperature`: Control the temperature adaptation process.
 #   *   `b_star_n_star`:  Corresponds to the `n*` parameter in the balance score formula, representing the desired number of correct responses per query.
  #  *   `current_step`: Tracks the training step for periodic temperature adaptation.
   # *   `adaptation_interval`: Determines how many steps to take before adapting the temperature.
    #*   `evaluation_set_size`:  Specifies the number of data points to collect for evaluating the balance score and adapting the temperature.
   # *   `exploration_data`: A list to store data used for temperature adaptation.

#2. **`adapt_temperature()` Method:**
#    *   This method implements the core logic of B-STAR's temperature adaptation.
#    *   It performs a grid search over a predefined set of temperatures (you can customize this set).
#    *   For each temperature:
#        *   It re-calculates the attention weights using the current `exploration_data` and the candidate temperature.
#        *   It computes the balance score for each sample in the `exploration_data`. The balance score calculation will need to be adjusted according to your specific definition of "correct" and "selected" states. I've provided a basic example, but you'll need to refine it.
#        *   It averages the balance scores across the `exploration_data`.
#    *   It selects the temperature that yields the highest average balance score.
#    *   It updates the `self.temperature` using the best temperature, applying the decay factor and ensuring it doesn't fall below `min_temperature`.
#    *   The balance score is a crucial element of B-STAR. The way I've implemented it here is a placeholder and needs to be tailored to your specific task and how you define the quality of generated responses. The formula from the paper is:

  #      ```
 #       balance_score = min(n_i' / n*, 1) * (n_i / n_i)
  #      ```

   #     Where:
    #    *   `n_i'` is the number of unique, correct responses.
   #     *   `n_i` is the total number of selected responses.
    #    *   `n*` is a hyperparameter ( `b_star_n_star` in the code).

#3. **`add_to_exploration_data()` Method:**
#    *   This is a helper method to accumulate data that will be used later by `adapt_temperature()`. You'll need to call this method during training in latent mode to store the relevant information (states, state qualities, and rewards).

#4. **Integration into `forward()`:**
 #   *   The `forward()` method now includes a check: `if self.exploration_data is not None and self.current_step % self.adaptation_interval == 0:`. This means temperature adaptation happens only in latent mode and only at intervals defined by `adaptation_interval`.
 #   *   The line `self.exploration_data = None` resets the buffer after adaptation.
 #   *   The temperature is used in the softmax calculation for attention weights: `attn_scores = attn_scores / self.temperature`.
 #   *   `self.current_step` is incremented at the end of each forward pass.

#**How to Use:**

#1. **Initialization:** Set the B-STAR parameters when creating an instance of `BinaryLatentTransformer`.
#2. **Training Loop (in latent mode):**
 #   *   Call `model.enable_latent_mode()` before entering the latent processing phase.
  #  *   During training in latent mode, periodically (e.g., after each batch or epoch), collect the necessary data:
#        *   `states`: The output of the RNN (before selection).
#        *   `state_qualities`: The output of the state evaluator.
 #       *   `rewards`:  The rewards obtained for each sample (you'll need to define how these are calculated based on your task).
 #       *   Call `model.add_to_exploration_data(states, state_qualities, rewards)` to store this data.
 #   *   The `forward()` method will automatically trigger temperature adaptation at the specified intervals.
#3. **Reward Mechanism:** You'll need a way to get rewards during training. This could be a separate reward model (as in the paper) or some other form of feedback that indicates the quality of the generated responses.
#4. **Balance Score Refinement:** The balance score calculation in `adapt_temperature()` needs to be carefully adjusted to match your specific task and the way you evaluate the correctness and selection of latent states.

#**Important Considerations:**

#*   **Computational Cost:** The temperature adaptation process does add some computational overhead because of the grid search and balance score calculations. However, it's done periodically, not at every step.
#*   **Hyperparameter Tuning:** The B-STAR parameters themselves might need some tuning (e.g., `initial_temperature`, `temperature_decay`, `min_temperature`, `adaptation_interval`, `evaluation_set_size`, `b_star_n_star`).
#*   **Exploration Data Size:** The `evaluation_set_size` determines how much data is used for temperature adaptation. You might need to experiment to find a good balance between adaptation accuracy and computational cost.

#This revised response provides a more complete and detailed implementation of B-STAR's exploration policy and temperature adaptation within your `BinaryLatentTransformer`. Remember to adapt the balance score calculation and the reward mechanism to your specific needs.
