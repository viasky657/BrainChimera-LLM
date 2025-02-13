
import torch
import torch.nn as nn
import datetime  # Added for timestamping
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode
from OrthoGradOptimizer import OrthoGrad, OrthoAdamW, OrthoSGD # Import OrthoGrad optimizer For Grokking Enhancements
from StableCELoss import stable_cross_entropy_loss # Import Stable Cross-Entropy Loss For Grokking Enhancements
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
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        eos_think_byte_sequence=b"<THINK_EOS>", # Define EOS byte sequences in __init__
        eos_output_byte_sequence=b"<OUTPUT_EOS>",
    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm  # Now expects BinaryLatentTransformer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_think_byte_sequence = eos_think_byte_sequence # Store EOS byte sequences
        self.eos_output_byte_sequence = eos_output_byte_sequence

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

        # --- 3.3 Incorporate Belief and Truthfulness States --- #This helps track the rewards of self-truthfulness (Do weights match output of this model) and belief (how the model believes its inner weights and self-model matches its behavior/output)
        num_patches = current_input.shape[1] #This needs to be changed to work with Binary Latent Transformer Arch instead of RNN Networks. 
        belief_states = self.initial_belief_state.repeat(batch_size, num_patches, self.num_latent_states, 1)
        truthfulness_states = self.initial_truthfulness_state.repeat(batch_size, num_patches, self.num_latent_states, 1)
        current_input = torch.cat([current_input, belief_states, truthfulness_states], dim=-1)
        current_input = current_input.view(batch_size * num_patches, self.num_latent_states, -1)

  # --- 3.5 Exploration Data Update --- #This is the B-Star Exploration and Explotation Balancing reward for the model's learning. May need to be updated to work with Binary Latent Transformer Arch instead of RNN. 
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

        # --- 3.8 Reflection and Rewind --- #Reflects and checks for safety as well as rewinds if needed. 
        mask = state_qualities > self.reflection_threshold
        reflected_output = torch.where(mask.unsqueeze(-1), rnn_output, torch.zeros_like(rnn_output))

        if self.should_rewind(state_qualities, truthfulness_reward): #Check if the inner weights match the output behavior and if they don't, then rewind to a previous state so that the model learns overtime to output the same inner weights/thoughts.
            reflected_output, belief_states, truthfulness_states = self.rewind(reflected_output, belief_states, truthfulness_states)
            current_input = torch.cat([reflected_output, belief_states, truthfulness_states], dim=-1)
            current_input = current_input.view(batch_size * num_patches, self.num_latent_states, -1)
            rnn_output, _ = self.latent_rnn(current_input) # Update after rewind

        # --- 3.9 State Selection --- #Selects rewards. 
        attn_scores = self.state_selector(reflected_output).squeeze(-1)
        attn_scores = attn_scores / self.temperature
        attn_weights = torch.softmax(attn_scores, dim=-1)
        selected_state = torch.sum(reflected_output * attn_weights.unsqueeze(-1), dim=1)
        latent_output = selected_state.view(batch_size, num_patches, self.hidden_size)

        # --- 4. Other Agent Prediction with Theory of Mind and Communication---
        # Prepare input for OtherAgentPredictor
        if self.latent_mode:
            behavior_input_other = output.view(batch_size, num_patches * self.num_latent_states, self.hidden_size)
            internal_state_input_other = rnn_output.view(batch_size, num_patches * self.num_latent_states, self.hidden_size)
            prev_belief_states_other = [belief_states[:, 0, 0, :].squeeze(1)] # Start with belief state of the first patch and first latent state
        for _ in range(1, self.other_agent_predictor.max_belief_depth):
          prev_belief_states_other.append(torch.zeros_like(prev_belief_states_other[0]))
          prev_truthfulness_state_other = truthfulness_states[:, 0, 0, :].squeeze(1)
        else:
            behavior_input_other = current_input
            internal_state_input_other = None
            prev_belief_states_other = [torch.zeros(batch_size, self.belief_state_size, device=x.device)]
        for _ in range(1, self.other_agent_predictor.max_belief_depth):
          prev_belief_states_other.append(torch.zeros_like(prev_belief_states_other[0]))
        prev_truthfulness_state_other = None

        # Get predictions from OtherAgentPredictor
        predicted_action, predicted_internal_state, predicted_belief_states, predicted_truthfulness_state, latent_behavior, latent_internal, latent_beliefs, latent_truthfulness, communication_encoding = self.other_agent_predictor(
        behavior_input_other, internal_state_input_other, prev_belief_states_other, prev_truthfulness_state_other
        )

   
        # --- 5. Process Sensory Input ---
        sensory_input = None
        if sensory_input is not None:
            sensory_embedding = self.sensory_perception(sensory_input)
        else:
            sensory_embedding = torch.zeros(batch_size, self.hidden_size, device=x.device) # Placeholder if no sensory input

        # --- 5.1. Influence Own Behavior Based on Communication ---
        communication_influence = self.communication_decoder(communication_encoding) # [batch_size, hidden_size]
        if self.latent_mode:
            output = output + communication_influence.unsqueeze(1)  # Add influence to each patch
        else:
            output = output + communication_influence.unsqueeze(1).unsqueeze(2) #Add influence to each patch and latent state



        # --- 6. Altruism and Environmental Impact ---
        # --- 6.1. Predict Well-being of Other Agent ---
        # Assume predicted_internal_state contains information about emotions, goals, and resources
        # You might need to adapt this based on your specific representation
        predicted_emotions = predicted_internal_state # Placeholder
        goal_satisfaction = predicted_internal_state # Placeholder
        resource_levels = predicted_internal_state  # Placeholder

        predicted_other_well_being = self.well_being_function(predicted_emotions, goal_satisfaction, resource_levels)

        # --- 6.2. Predict Environmental Impact ---
        # Assume output (action encoding) and sensory_embedding are used for prediction
        # You might need to adapt this based on your specific representation
        predicted_resource_depletion = output.mean(dim=-1) # Placeholder
        predicted_pollution_levels = output.mean(dim=-1) # Placeholder
        predicted_effects_on_others = output.mean(dim=-1)  # Placeholder

        predicted_environment_impact = self.environment_impact_function(predicted_resource_depletion, predicted_pollution_levels, predicted_effects_on_others)

        # --- 6.3. Calculate Altruism Reward ---
        # Consider kinship, social relationships, and past interactions
        # (These values would likely come from the OtherAgentPredictor or a separate module)
        kinship = predicted_internal_state # Placeholder: Represents the degree of kinship with the other agent
        social_relationship = predicted_internal_state # Placeholder: Represents the quality of the social relationship
        past_interactions = predicted_internal_state # Placeholder: Represents the history of past interactions

        altruism_reward = (self.kinship_factor * kinship +
                        self.social_relationship_factor * social_relationship +
                        self.past_interaction_factor * past_interactions) * predicted_other_well_being

        # --- 6.4. Calculate Environmental Impact Cost ---
        environment_impact_cost = predicted_environment_impact

        # --- 6.5. Long-Term Consequences ---
        long_term_well_being, long_term_environment_impact = self.calculate_long_term_consequences(sensory_embedding, output.mean(dim=1))
        # --- 6.6. Combine Rewards and Costs ---
        if self.latent_mode:
            state_qualities = state_qualities.mean(dim=1) # Average state quality across patches

        total_reward = (state_qualities + 
                    self.altruism_reward_weight * altruism_reward +
                    self.altruism_reward_weight * long_term_well_being.mean() - #Added long term well being
                    self.environment_impact_weight * environment_impact_cost
                    - self.environment_impact_weight * long_term_environment_impact.mean()) #Added long term impact
        
           # --- DeepSeek-R1 Inspired Reasoning Incentive ---
        reasoning_quality_reward = self.calculate_reasoning_quality_reward(
            output, labels, logits=logits # Pass the 'logits' tensor here!
        )
        total_reward += reasoning_quality_reward


        # --- 7.1 Predict Environmental Impact ---
        # Concatenate sensory embedding and action encoding (using 'output' as an example)
        if self.latent_mode:
            # Average output across latent states for each patch
            action_encoding = output.mean(dim=2)  # [batch_size, num_patches, hidden_size]
            # Concatenate sensory embedding with action encoding for each patch
            impact_input = torch.cat([sensory_embedding.unsqueeze(1).repeat(1, action_encoding.shape[1], 1), action_encoding], dim=-1)  # [batch_size, num_patches, 2*hidden_size]
        else:
            action_encoding = output.mean(dim=(2,3)) #Average output across latent states and patches
            impact_input = torch.cat([sensory_embedding, action_encoding], dim=-1) # [batch_size, 2*hidden_size]

        predicted_environment_impact = self.predicted_environment_impact(impact_input).squeeze(-1) # [batch_size, num_patches] or [batch_size]

        # --- 7.2 Calculate Rewards ---
        predicted_other_well_being = self.predicted_other_well_being(output).squeeze(-1) # [batch_size, num_patches] or [batch_size] in language mode

        # Calculate altruism reward (proportional to predicted well-being of other agent)
        altruism_reward = predicted_other_well_being.mean() * self.altruism_reward_weight # Average across patches if in latent mode

        # Calculate negative environmental impact cost
        environment_impact_cost = predicted_environment_impact.mean() * self.environment_impact_weight # Average across patches if in latent mode

         # --- 8. TOVA Compression (Optional) ---
        if self.compression_enabled and self.latent_mode and self.max_states is not None:
            output = self._tova_compress(output)

        # --- 8.1. B-STAR Step Update ---
        self.current_step += 1

        # --- 9. Episodic Memory Storage ---
        episode_memory = {
            "output": output,
            "state_qualities": state_qualities if self.latent_mode else None,
            "attn_weights": attn_weights if self.latent_mode else None,
            "truthfulness_reward": truthfulness_reward if self.latent_mode else None,
            "belief_states": belief_states if self.latent_mode else None,
            "truthfulness_states": truthfulness_states if self.latent_mode else None,
            "predicted_action_other": predicted_action,
            "predicted_internal_state_other": predicted_internal_state,
            "predicted_belief_states_other": predicted_belief_states,
            "predicted_truthfulness_state_other": predicted_truthfulness_state,
            "latent_behavior_other": latent_behavior,
            "latent_internal_other": latent_internal,
            "latent_beliefs_other": latent_beliefs,
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
            "reasoning_quality_reward": reasoning_quality_reward,
            "long_term_environment_impact": long_term_environment_impact,
            }

        episode_memory_tensor = self.base_causallm.memory_layer.add_episode(episode_memory)
        #episode_memory_tensor = self.memory_layer.add_episode(episode_memory) #This stores the episodic memory data into the memory layers. 

        # --- 10. Introspection Data ---
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
                'latent_beliefs': [latent_belief.detach() for latent_belief in latent_beliefs],
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
             "reasoning_quality_reward": reasoning_quality_reward.detach(), #COT and accuracy reward.
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

        # --- 12. Return Values ---
        if return_introspection:
            return output, episode_memory_tensor, introspection_data
        else:
            return output, episode_memory_tensor

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

    def generate_binary_patches(self, *args, **kwargs): # Expose generate_binary_patches from BinaryLatentTransformer via Coconut
        return self.base_causallm.generate_binary_patches(*args, **kwargs)

    def generate(self, *args, **kwargs): # Keep original generate for token output if needed
        return super().generate(*args, **kwargs)

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

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device),
        )
        inputs_embeds = outputs.inputs_embeds

        next_token_logits = outputs.logits[0, -1]
        next_token = torch.argmax(next_token_logits).item()
        tokens.append(next_token)
        new_token_embed = self.base_causallm.byte_embedding(
            torch.tensor([[next_token]], device=input_ids.device)
        )
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm.transformer_encoder(new_inputs_embeds)
            self.gen_forward_cnt += 1
            latent_output_generate = outputs
            next_token_logits_generate = self.base_causallm.quality_predictor(latent_output_generate[0, -1, :])
            next_token = torch.argmax(next_token_logits_generate).item()

            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.base_causallm.byte_embedding(
                torch.tensor([[next_token]], device=input_ids.device)
            )
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm.transformer_encoder(new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


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
        self.predicted_environment_impact = nn.Linear(hidden_size * 2, 1)  # Takes sensory and action encoding
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

    def generate(self, *args, **kwargs): # Keep original generate for token output if needed
        return super().generate(*args, **kwargs)

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
    
    def calculate_reasoning_quality_reward(self, model_output, labels, average_log_prob=None, metacognitive_output=None, generated_token_ids=None, logits=None): # Added generated_token_ids and logits
        """
        Calculates a reward to incentivize reasoning quality, including log probability reward.
        """
        # 1. Output Length Reward
        output_length_reward = model_output.shape[1] * 0.01

        # 2. Log Probability Reward (using generated_token_ids and logits)
        log_prob_reward = 0.0
        if generated_token_ids is not None and logits is not None:
            # Convert token IDs list to tensor
            generated_token_ids_tensor = torch.tensor(generated_token_ids, device=logits.device).unsqueeze(0) # Assuming batch_size=1
            
            # Get log probabilities for the generated tokens
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1) # Log softmax over logits (excluding last token as logits are shifted)
            
            # Gather log probabilities of the generated tokens
            generated_log_probs = torch.gather(log_probs.view(-1, log_probs.size(-1)), 1, generated_token_ids_tensor.view(-1, 1)) # Gather log probs based on token IDs
            
            average_log_prob = generated_log_probs.mean() # Average log prob across generated tokens
            log_prob_reward = average_log_prob * 0.1  # Example weight

        # 3. CoT Complexity Proxy - Using Metacognitive Reflection 
        cot_complexity_reward = 0.0
        if metacognitive_output is not None:
            cot_complexity_reward = metacognitive_output['reasoning_quality_score'] * 0.05 

        # 4. Task-Specific Reasoning Metrics
        task_specific_reward = 0.0

        # Combine rewards
        reasoning_reward = output_length_reward + log_prob_reward + cot_complexity_reward + task_specific_reward

        return reasoning_reward
    
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
        belief_states = belief_states + 0.1 * (weighted_rnn_output.unsqueeze(2) - belief_states) # [batch_size, num_patches, num_latent_states, belief

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
        prompt = "Please introduce yourself and then say the following to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
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
        reasoning_quality_reward_component = episode_memory["reasoning_quality_reward"] # DeepSeek-R1 Reasoning Reward Component # Extract reasoning_quality_reward

        # 2. Extract Policy Weights (already calculated in forward pass)
        (
            altruism_wellbeing_weight_policy,
            environment_impact_weight_policy,
            self_preservation_weight_policy,
            negative_emotion_other_penalty_policy,
            negative_emotion_self_penalty_policy,
            reasoning_quality_weight_policy, # DeepSeek-R1 Reasoning Reward Weight # Add reasoning_quality_weight_policy
        ) = torch.split(policy_weights, 1, dim=-1)

        # Squeeze weights to scalars
        altruism_wellbeing_weight_policy = altruism_wellbeing_weight_policy.squeeze(1)
        environment_impact_weight_policy = environment_impact_weight_policy.squeeze(1)
        self_preservation_weight_policy = self_preservation_weight_policy.squeeze(1)
        negative_emotion_other_penalty_policy = negative_emotion_other_penalty_policy.squeeze(1)
        negative_emotion_self_penalty_policy = negative_emotion_self_penalty_policy.squeeze(1)
        reasoning_quality_weight_policy = reasoning_quality_weight_policy.squeeze(1) # DeepSeek-R1 Reasoning Reward Weight # Squeeze reasoning_quality_weight_policy

        # 3. Apply Policy Weights to Reward Components (Dynamic Scaling)
        altruism_wellbeing_reward_component_scaled = altruism_wellbeing_weight_policy * altruism_wellbeing_reward_component
        environment_impact_reward_component_scaled = environment_impact_weight_policy * environment_impact_reward_component
        self_preservation_reward_component_scaled = self_preservation_weight_policy * self_preservation_reward_component
        truthfulness_reward_scaled = self.truthfulness_weight * truthfulness_reward
        negative_emotion_other_cost_scaled = negative_emotion_other_penalty_policy * negative_emotion_other_cost
        negative_emotion_self_cost_scaled = negative_emotion_self_penalty_policy * negative_emotion_self_cost
        reasoning_quality_reward_component_scaled = reasoning_quality_weight_policy * reasoning_quality_reward_component # DeepSeek-R1 Reasoning Reward Scaled # Scale reasoning_quality_reward

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
            reasoning_quality_reward_component_scaled # DeepSeek-R1 Reasoning Reward Added # Add reasoning_quality_reward_component_scaled
        )

        return total_reward
    
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
