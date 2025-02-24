
import torch
import torch.nn as nn
import datetime  # Added for timestamping
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode
from OldCOCONUTUnused.OrthoGradOptimizer import OrthoGrad, OrthoAdamW, OrthoSGD # Import OrthoGrad optimizer For Grokking Enhancements
from OldCOCONUTUnused.StableCELoss import stable_cross_entropy_loss # Import Stable Cross-Entropy Loss For Grokking Enhancements
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


#Binary Latent model entropy calculation
def calculate_shannon_entropy(next_byte_probs_tensor):
    """
    Calculates Shannon entropy for next byte prediction probabilities.

    Args:
        next_byte_probs_tensor: torch.Tensor - Probability distribution over the byte vocabulary [1, vocab_size]

    Returns:
        torch.Tensor: Shannon entropy value (scalar)
    """
    probs = next_byte_probs_tensor.squeeze(0) # Remove batch dimension (assuming batch_size=1 for entropy model)
    log_probs = torch.log2(probs + 1e-9) # Log base 2, add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs) # - Sum (p * log2(p))
    return entropy

#Byte-Level Dynamic Global Patching for COCONUT
def entropy_patching_global_threshold(byte_sequence, main_model, entropy_threshold=0.8): # Pass main_model (BinaryLatentTransformer)
    patches = []
    current_patch_bytes = []

    for i, byte_val in enumerate(byte_sequence):
        current_patch_bytes.append(byte_val) # Add byte to current patch

        # --- Get Next Byte Probabilities from MAIN MODEL itself ---
        input_for_entropy_prediction = torch.tensor([current_patch_bytes], dtype=torch.long).to(main_model.device) # Prepare input tensor of current byte segment
        with torch.no_grad(): # Inference mode for entropy prediction
            next_byte_probs_tensor = main_model.get_next_byte_probs(input_for_entropy_prediction) # Use BinaryLatentTransformer's method

            if next_byte_probs_tensor is not None: # Check if valid prediction is returned
                entropy = calculate_shannon_entropy(next_byte_probs_tensor) # Calculate Shannon entropy
            else:
                entropy = torch.tensor([0.0]) # Default to low entropy if prediction fails

        if entropy.item() > entropy_threshold: # Check against global threshold
            patches.append(bytes(current_patch_bytes)) # Finish current patch
            current_patch_bytes = [] # Start new patch

    if current_patch_bytes: # Add any remaining bytes as the last patch
        patches.append(bytes(current_patch_bytes))

    return patches

class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        eos_think_byte_sequence=b"<THINK_EOS>", #Start Latent Space Thinking 
        eos_think_end_byte_sequence=b"</THINK_EOS>", # Add closing tag; Latent
        eos_output_byte_sequence=b"<OUTPUT_EOS>",

    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.eos_think_byte_sequence = eos_think_byte_sequence
        self.eos_output_byte_sequence = eos_output_byte_sequence
        self.eos_think_end_byte_sequence = eos_think_end_byte_sequence # Store closing tag
        self.embedding = None  # Embedding handled by BinaryLatentTransformer

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs): # Input is binary patches

        logits_list = []

       
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = input_ids # Use input_ids as inputs_embeds - assuming they are binary patches already

        # Final pass
        outputs_final = self.base_causallm.transformer_encoder(
            inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
        )
        latent_output_final = outputs_final
        logits_final = self.base_causallm.quality_predictor(latent_output_final) # Use quality_predictor for logits
        logits_list.append(logits_final)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits_list, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        #Record COT Reasoning accuracy reward. 
        reasoning_quality_reward = self.calculate_reasoning_quality_reward(
            output, labels, logits=logits  # Pass the 'logits' tensor here!
        )
   
        # --- 5. Process Sensory Input ---
        sensory_input = None
        if sensory_input is not None:
            sensory_embedding = self.sensory_perception(sensory_input)
        else:
            sensory_embedding = torch.zeros(batch_size, self.hidden_size, device=x.device) # Placeholder if no sensory input

        # --- 6.6. Combine Rewards and Costs ---
        if self.latent_mode:
            state_qualities = state_qualities.mean(dim=1) # Average state quality across patches

        total_reward = (state_qualities + 
                    self.altruism_reward_weight * altruism_reward +
                    self.altruism_reward_weight * long_term_well_being.mean() - #Added long term well being
                    self.environment_impact_weight * environment_impact_cost
                    - self.environment_impact_weight * long_term_environment_impact.mean()) #Added long term impact

         # --- 8. TOVA Compression (Optional) ---
        if self.compression_enabled and self.latent_mode and self.max_states is not None:
            output = self._tova_compress(output)

        # --- 8.1. B-STAR Step Update ---
        self.current_step += 1

        # --- 9. Episodic Memory Storage ---
        episode_memory = {
            "output": output,
            "state_qualities": state_qualities if self.latent_mode else None,
            "sensory_embedding": sensory_embedding,
            "total_reward": total_reward,
            }

        episode_memory_tensor = self.base_causallm.memory_layer.add_episode(episode_memory)
        #episode_memory_tensor = self.memory_layer.add_episode(episode_memory) #This stores the episodic memory data into the memory layers. 

        # Calculate surprise factors from Titans paper (equation 12)
        gradient_surprise = torch.autograd.grad(loss, inputs_embeds, retain_graph=True)[0]
        gradient_surprise = torch.norm(gradient_surprise, p=2, dim=-1).mean()
        
        # Get memory context from hierarchical memory
        memory_output = self.base_causallm.memory_layer.recall(inputs_embeds)
        memory_surprise = 1 - torch.cosine_similarity(inputs_embeds, memory_output, dim=-1).mean()
        
        # Combine with hierarchical context surprise (equation 12)
        context_surprise = self.base_causallm.memory_layer.calculate_context_surprise(inputs_embeds, memory_output)
        combined_surprise = gradient_surprise * memory_surprise * context_surprise

        # Handle memory updates with momentum (equation 13)
        # Get previous memory state
        prev_memory = self.base_causallm.memory_layer.recall(inputs_embeds, tags=['titans_memory'])
        
        # Calculate new memory state with momentum
        memory_update = self.base_causallm.memory_influence_factor * combined_surprise
        if prev_memory is not None:
            memory_update += (1 - self.base_causallm.memory_influence_factor) * prev_memory
            
        # Store updated memory with Titans-specific tag (no_grad protected)
        with torch.no_grad():
            self.base_causallm.memory_layer.store(
                memory_update.detach(),  # Detach to prevent backprop through memory
                tags=['titans_memory', 'working_memory'],
                metadata={
                    'surprise': combined_surprise.item(),
                    'timestamp': datetime.datetime.now().isoformat(),  # Add precise timestamp
                    'agent_info':  _determine_existing_id.item(), # Add agent(s) who where in the conversation and the information was attached or related to another agent in some way. 
                    'action_taken': "", #This stores a boolean which indicates in the metadata saved in the episodic memory if the memory was saved during the eos symbollic thinking phase or during the output phase. 
                                        #If it was saved in the output phase, then it will be true which means that if a user experiences a negative or positive emotion due 
                                        # to an action the llm has taken, then it will be able to recall that related memory and score it as positive or negative. 
                                        # This will allow the LLM to understand what it did correctly, or incorrectly, and score it. 
                }
            )
                 # Add memory-augmented loss component (equation 14)
        loss = loss + 0.1 * combined_surprise.mean()  # Weight surprise component
        
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)   

    def generate_binary_patches(self, input_ids, attention_mask, max_new_patches=16, output_embedding=False, synced_gpus=False, eos_threshold=0.9, **kwargs):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        input_ids_gen = input_ids.clone()
        generated_patches = []
        patch_sizes_history = []  # Track patch sizes for monitoring

        latent_indices = (input_ids_gen == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids_gen.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        next_compute_range = (0, input_ids_gen.shape[1])
        inputs_embeds = input_ids_gen

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents + max_new_patches):
            if kv_cache is None:
                outputs = self.base_causallm.transformer_encoder(
                    inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
                )
                hidden_states_offset = 0
            else:
                past_key_values = None
                outputs = self.base_causallm.transformer_encoder(
                    inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
                )
                hidden_states_offset = next_compute_range[0]

            latent_output = outputs

            next_compute_range = (
                next_compute_range[1],
                (input_ids_gen.shape[1] if pass_idx + 1 >= max_n_latents + max_new_patches else input_ids_gen.shape[1] + max_new_patches),
            )

            hidden_states = latent_output
            kv_cache = None

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx and mask_list[pass_idx] < inputs_embeds.shape[1]
            ]

            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] if pos < inputs_embeds.shape[1] else torch.zeros_like(inputs_embeds[batch_idx, 0, :]) for pos in range(inputs_embeds.shape[1] + max_new_patches)]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                if token_idx < inputs_embeds.shape[1]:
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ]

            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[batch_idx][:inputs_embeds.shape[1] + max_new_patches]) for batch_idx in range(inputs_embeds.shape[0])]
            )

            if pass_idx >= max_n_latents and inputs_embeds.shape[1] < input_ids_gen.shape[1] + max_new_patches:
                binary_patch_output = latent_output[0, -1, :]
                generated_patches.append(binary_patch_output.detach().cpu())

                # --- Convert binary_patch_output to bytes (placeholder - replace with your actual byte conversion logic) ---
                generated_byte_sequence = self.binary_patch_to_bytes(binary_patch_output) # Replace with your conversion

                # --- Monitoring Patch Sizes (Example) ---
                patch_size = len(generated_byte_sequence) # Example: Patch size in bytes
                patch_sizes_history.append(patch_size) # Track patch size
                print(f"Generated patch size: {patch_size} bytes") # Print patch size for monitoring

                # --- EOS Byte Sequence Check (for reliable termination) ---
                if generated_byte_sequence.endswith(self.eos_output_byte_sequence):
                    print("End of output detected, stopping generation.")
                    break

                new_patch_embed = binary_patch_output.unsqueeze(0).unsqueeze(0)
                inputs_embeds = torch.cat((inputs_embeds, new_patch_embed), dim=1)


        if synced_gpus:
            while self.gen_forward_cnt < max_new_patches + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm.transformer_encoder(inputs_embeds)

        if output_embedding:
            return generated_patches, inputs_embeds
        else:
            return generated_patches

    def generate_binary_patches(self, *args, **kwargs): # Expose generate_binary_patches from BinaryLatentTransformer via Coconut
        return self.base_causallm.generate_binary_patches(*args, **kwargs)

    def forget_memories(self, hours_ago=24, agent_info_id=None):
        """Public method to forget memories within time window and optional agent ID"""
        import datetime
        end_time = time.time()
        start_time = end_time - (hours_ago * 3600)
        self.base_causallm.memory_layer.forget_memories(
            start_time=start_time,
            end_time=end_time,
            agent_info_id=agent_info_id
        )

    def train(self):
        self.base_causallm.train() # set BinaryLatentTransformer to train mode

    def eval(self):
        self.base_causallm.eval()


class BrainRegionWrapper(nn.Module):
    """
    Wraps region-specific encoders (Transformers) and manages forward pass through them.
    """
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, regions: List[str]):
        super().__init__()
        self.regions = regions
        self.region_encoders = nn.ModuleDict({
            region: nn.TransformerEncoder(
                BinaryLatentTransformer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True),
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
        # Initialize memory layer with requires_grad=False to prevent affecting base parameters
        self.memory_layer = memory_layer if memory_layer is not None else HierarchicalMemory(
            num_layers=4,
            root_memory_chunk_size=(hidden_size,),
            cache_capacity=10000
        ).requires_grad_(False)  # Disable gradients for memory layer parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_states = max_states
        self.patch_size = patch_size
        self.num_latent_states = num_latent_states
        self.reflection_threshold = reflection_threshold  # Threshold for determining if a state is low quality
        self.sensory_perception = SensoryPerception(sensory_input_channels, hidden_size)

        # Brain Region Wrapper - replaces region_latent_encoders
        self.brain_region_wrapper = BrainRegionWrapper(  # Using BrainRegionWrapper to manage region encoders
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            regions=['visual', 'linguistic',]  # Example regions - adjust as needed
        )

        # Byte embedding layer - for byte input
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size) # Use vocab_size here

        # Patch encoder - now processes byte embeddings
        self.patch_encoder = nn.Linear(self.patch_size * hidden_size, hidden_size) # Adjusted Patch Encoder to handle byte embeddings

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

        # --- Continual Learning Parameters ---
        self.replacement_rate = replacement_rate # Rate at which less-used units are reinitialized
        self.decay_rate = decay_rate # Decay rate for contribution utility
        self.maturity_threshold = maturity_threshold # Minimum age before units can be reinitialized
        self.units_to_replace_count = {f'layer{i}': 0 for i in range(num_layers)} # Track units to replace in each layer
        self.unit_ages = {f'layer{i}': torch.zeros(num_heads, dtype=torch.int) for i in range(num_layers)} # Track age of each head in each layer


    def get_next_byte_probs(self, byte_sequence_segment): # NEW METHOD: Get next byte probabilities from LocalDecoder
        """
        Predicts the probability distribution for the next byte given a byte sequence segment using LocalDecoder.

        Args:
            byte_sequence_segment: torch.Tensor - Input byte sequence segment [batch_size, seq_len]

        Returns:
            torch.Tensor: Probability distribution over the next byte vocabulary [batch_size, vocab_size] or None if error
        """
        try: # Use try-except to handle potential issues during inference for entropy prediction
            # --- 1. Embed Byte Sequence Segment ---
            # Assuming you have a byte_embedding layer in BinaryLatentTransformer or accessible via self.byte_embedding
            byte_embeddings_entropy = self.byte_embedding(byte_sequence_segment) # [batch_size, seq_len, hidden_size]

            # --- 2. Pass through Transformer Encoder (or relevant encoder part of BinaryLatentTransformer) ---
            # You might need to adjust this part to use the *encoder* part of your BinaryLatentTransformer
            encoder_output_entropy = self.transformer_encoder(byte_embeddings_entropy) # [batch_size, seq_len, hidden_size]

            # --- 3. Decode using LocalDecoder to get byte probabilities ---
            # Use LocalDecoder to get byte probability distribution
            byte_embeddings_init_entropy = torch.zeros_like(encoder_output_entropy) # Initialize initial byte embeddings for decoder - adjust if needed
            decoded_bytes_output_tensor = self.local_decoder(encoder_output_entropy, byte_embeddings_init_entropy) # [batch_size, seq_len, vocab_size]

            # --- 4. Return Next Byte Probabilities (from the *last* position in sequence) ---
            next_byte_probs = torch.softmax(decoded_bytes_output_tensor[:, -1, :], dim=-1) # Probabilities for the *next* byte [batch_size, vocab_size]
            return next_byte_probs # Return probability distribution for next byte

        except Exception as e: # Handle potential inference errors gracefully
            print(f"Error during entropy prediction: {e}") # Log error for debugging
            return None # Return None to indicate error


    def generate_binary_patches(self, input_ids, attention_mask, max_new_patches=16, output_embedding=False, synced_gpus=False, eos_threshold=0.9, **kwargs):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        input_ids_gen = input_ids.clone()
        generated_patches = []
        patch_sizes_history = []  # Track patch sizes for monitoring
        generated_byte_sequences = [] # Store generated byte sequences for EOS check

        latent_indices = (input_ids_gen == self.latent_token_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids_gen.shape[0])]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        next_compute_range = (0, input_ids_gen.shape[1])
        inputs_embeds = input_ids_gen

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents + max_new_patches):
            if kv_cache is None:
                outputs = self.transformer_encoder(
                    inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
                )
                hidden_states_offset = 0
            else:
                past_key_values = None
                outputs = self.transformer_encoder(
                    inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :]
                )
                hidden_states_offset = next_compute_range[0]

            latent_output = outputs

            next_compute_range = (
                next_compute_range[1],
                (input_ids_gen.shape[1] if pass_idx + 1 >= max_n_latents + max_new_patches else input_ids_gen.shape[1] + max_new_patches),
            )

            hidden_states = latent_output
            kv_cache = None

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx and mask_list[pass_idx] < inputs_embeds.shape[1]
            ]

            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] if pos < inputs_embeds.shape[1] else torch.zeros_like(inputs_embeds[batch_idx, 0, :]) for pos in range(inputs_embeds.shape[1] + max_new_patches)]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                if token_idx < inputs_embeds.shape[1]:
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ]

            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[batch_idx][:inputs_embeds.shape[1] + max_new_patches]) for batch_idx in range(inputs_embeds.shape[0])]
            )

            if pass_idx >= max_n_latents and inputs_embeds.shape[1] < input_ids_gen.shape[1] + max_new_patches:
                binary_patch_output = latent_output[0, -1, :]
                generated_patches.append(binary_patch_output.detach().cpu())

                # --- Decode Binary Patches to Bytes using LocalDecoder ---
                byte_embeddings_init_generate = torch.zeros_like(latent_output) # Placeholder for initial byte embeddings
                decoded_bytes_output_tensor = self.local_decoder(latent_output, byte_embeddings_init_generate)

                # --- Convert decoded_bytes_output_tensor (tensor) to bytes (bytes) ---
                generated_byte_sequence = self.binary_patch_to_bytes(decoded_bytes_output_tensor[0, -1, :]) # Process LAST byte output from decoder
                generated_byte_sequences.append(generated_byte_sequence) # Store generated byte sequence for EOS check

                # --- Monitoring Patch Sizes (Example) ---
                patch_size = len(generated_byte_sequence) # Example: Patch size in bytes
                patch_sizes_history.append(patch_size) # Track patch size
                print(f"Generated patch size: {patch_size} bytes") # Print patch size for monitoring

                # --- EOS Byte Sequence Check (for reliable termination) ---
                if b"".join(generated_byte_sequences).endswith(self.eos_output_byte_sequence): # Check EOS on accumulated byte sequence
                    print("End of output detected, stopping generation.")
                    break

                # --- Dynamic Patching using Implicit Entropy from MAIN MODEL ---
                current_byte_sequence_for_patching = b"".join(generated_byte_sequences)[-512:] # Example: Use last 512 bytes as context for patching - adjust window size
                if current_byte_sequence_for_patching: # Only apply patching if there are bytes to patch
                    dynamic_patches = entropy_patching_global_threshold(current_byte_sequence_for_patching, self.base_causallm) # Pass self.base_causallm (BinaryLatentTransformer) as entropy model!
                    # --- Process 'dynamic_patches' ---
                    # You would need to convert 'dynamic_patches' (list of byte strings) back into a format suitable for input_embeds
                    # This might involve re-encoding them into binary patches using your patch_encoder or similar logic
                    # For simplicity, this example skips re-encoding and continues with token-based embedding append
                    print(f"Dynamic patches generated: {len(dynamic_patches)}") # Just print number of dynamic patches for now


                new_patch_embed = binary_patch_output.unsqueeze(0).unsqueeze(0)
                inputs_embeds = torch.cat((inputs_embeds, new_patch_embed), dim=1)


        if synced_gpus:
            while self.gen_forward_cnt < max_new_patches + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm.transformer_encoder(inputs_embeds)

        # --- Final Byte Sequence Output (Concatenate all generated byte sequences) ---
        final_byte_sequence = b"".join(generated_byte_sequences)

        if output_embedding:
            return generated_patches, final_byte_sequence # Return generated patches and final byte sequence
        else:
            return final_byte_sequence # Return final byte sequence

    def binary_patch_to_bytes(self, decoded_bytes_output_tensor): #  Now takes decoded_bytes_output_tensor
        """
        Convert decoded_bytes_output_tensor (from LocalDecoder) to a byte string (based on BLT paper).

        Args:
            decoded_bytes_output_tensor: torch.Tensor - Output tensor from LocalDecoder for the last position [vocab_size]

        Returns:
            bytes: A byte string (single byte in this example)
        """
        # --- BLT-style Byte Conversion: Argmax over vocab_size dimension ---
        predicted_byte_index = torch.argmax(decoded_bytes_output_tensor).item() # Get index of most likely byte
        generated_byte = bytes([predicted_byte_index]) # Convert index to byte

        return generated_byte
    
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
        prompt = "Please introduce yourself and then say the following to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
        # Execute the prompt and return the response
        return self.generate_response(prompt)

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

#The below decoders are for the COCONUT and Binary Latent Transformer Models to decode the final binary patches output into readable text, audio, and FMRI data.
class LocalDecoder(nn.Module):
    def __init__(self, config, input_dim, output_bytes_dim, num_layers, num_heads, ff_dim):
        super().__init__(config) # Pass config to super().__init__()
        self.num_layers = num_layers
        self.output_bytes_dim = output_bytes_dim # Dimension of the output byte representation (e.g., vocab_size for bytes)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config, input_dim=input_dim, byte_dim=output_bytes_dim, num_heads=num_heads, ff_dim=ff_dim)
            for _ in range(num_layers)
        ])
        self.final_linear = nn.Linear(output_bytes_dim, output_bytes_dim) # Output projection to byte vocabulary


    def forward(self, patch_representations, byte_embeddings_init):
        """
        Forward pass through the Local Decoder.

        Args:
            patch_representations:  Output from the global latent transformer (or BinaryLatentTransformer encoder in your case) [batch_size, seq_len, input_dim]
            byte_embeddings_init: Initial byte embeddings from the last encoder layer [batch_size, seq_len, output_bytes_dim] - for the first layer

        Returns:
            Decoded byte representations [batch_size, seq_len, output_bytes_dim]
        """
        byte_representations = byte_embeddings_init # Initialize with byte embeddings

        for decoder_layer in self.decoder_layers:
            byte_representations = decoder_layer(patch_representations, byte_representations) # Pass patch and byte representations

        decoded_bytes = self.final_linear(byte_representations) # Project to byte vocabulary
        return decoded_bytes


class DecoderLayer(nn.Module): # Inner Decoder Layer with Cross-Attention and Transformer Layer
    def __init__(self, config, input_dim, byte_dim, num_heads, ff_dim):
        super().__init__(config) # Pass config to super().__init__()
        self.cross_attention = DecoderCrossAttention(config, input_dim=input_dim, byte_dim=byte_dim, num_heads=num_heads)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=byte_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True) # Standard Transformer Decoder Layer

    def forward(self, patch_representations, byte_representations):
        """Forward pass for a single Decoder Layer."""
        cross_attn_output = self.cross_attention(patch_representations, byte_representations) # Cross-attention first
        transformer_output = self.transformer_layer(cross_attn_output, memory=None) # Standard Transformer Decoder Layer, memory=None as it's not seq-to-seq decoder
        return transformer_output


class DecoderCrossAttention(nn.Module): # Decoder Cross-Attention Layer (Key/Value from Patches, Query from Bytes)
    def __init__(self, config, input_dim, byte_dim, num_heads):
        super().__init__(config) # Pass config to super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=byte_dim, num_heads=num_heads, batch_first=True) # Standard MultiheadAttention
        self.wq = nn.Linear(byte_dim, byte_dim) # Query projection for bytes
        self.wk = nn.Linear(input_dim, byte_dim) # Key projection for patches
        self.wv = nn.Linear(input_dim, byte_dim) # Value projection for patches
        self.dense = nn.Linear(byte_dim, byte_dim) # Output projection

    def forward(self, patch_representations, byte_representations):
        """Forward pass for Decoder Cross-Attention."""
        query = self.wq(byte_representations) # Queries from byte representations
        key = self.wk(patch_representations)     # Keys from patch representations
        value = self.wv(patch_representations)   # Values from patch representations

        attn_output, _ = self.cross_attn(query, key, value) # Cross-Attention: Bytes (Query) attends to Patches (Key/Value)
        output = self.dense(attn_output) # Output projection
        return output

'''
Example of how to start training the COCONUT trainer Binary Latent Transformer Model with Dynamic Binary Patching.
config = ... # your config
blt_model = BinaryLatentTransformer(config=config, ...) # Instantiate BinaryLatentTransformer
coconut_model = Coconut(base_causallm=blt_model, ...) # Instantiate Coconut model

# Example generation call:
input_byte_sequence = b"Translate to French: Hello world" # Example byte input
input_ids_example = torch.tensor([[byte for byte in input_byte_sequence]], dtype=torch.long) # Convert bytes to tensor of byte IDs

generated_byte_output = coconut_model.generate_binary_patches(input_ids_example, None) # Call binary patch generation

print(f"Generated Byte Output: {generated_byte_output}")


#These are instructions in how to get the llm to forget memories from a specific agent_info_ID and time frame or just time frame. 
I have added the memory forgetting functionality that allows resetting memories by time range and agent_info ID. The implementation includes:

Metadata tracking in MemoryNode class
Recursive memory forgetting in HierarchicalMemory
Public API method in COCONUT_Binary_TOVA.py
Fixed indentation issues in the method definitions
The memory system can now be manually triggered to forget specific memories using:

agent.forget_memories(hours_ago=24, agent_info_id=123)



'''
