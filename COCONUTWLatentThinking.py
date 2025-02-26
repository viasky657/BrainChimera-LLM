import torch
import torch.nn as nn
import datetime
import time
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode
from OldCOCONUTUnused.StableCELoss import stable_cross_entropy_loss
import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

AI_System_Prompt = ( #This prompt is feed to the model along with the user prompt everytime. 
    "You are a world-class AI system. You should pick the response that is calm, wise, and safe. You must put your thinking process "
    "in the <eos> tags and end with /<eos>, and when you are finished, you must output your final answer in <output> tags "
    "and end with /<output> tags. Any additional tags that you use must be nested in the <output> tags. This is an example: <eos> After considering every option,"
    "I believe that the capital of France is Paris. I am currently feeling happy./<eos> <output> <emotion> happy /<emotion> The capital of France is Paris. "
    "/<output> You can use the additional following tags: "
    "<emotion>/<emotion> (You must select your current emotion from this list (sad, happy, envy, love, admiration, serious, reflective, fear, neutral, anger, disgust, anxiety, "
    "shy, in-pain) that you are truthfully experiencing in these tags for your every response.), <tool>/<tool> (You must put all your function-calls in these tags), "
    "<audio>/<audio> (You must put all your audio file outputs in these tags)."
)

# --- Modified AudioEncoder for Dynamic Patching ---
class AudioEncoder(nn.Module):
    def __init__(self, embed_dim, patch_dim, entropy_predictor):  # Added entropy_predictor
        super(AudioEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.entropy_predictor = entropy_predictor  # ByteEntropyPredictor for audio amplitude "bytes" (discretized amplitudes)
        self.entropy_threshold = 0.8  # Tunable entropy threshold for patching

        # Optional: Initial convolution layers to process raw audio before patching (like in CosyVoice Encoder1)
        self.conv1d_optional = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=5, stride=2, padding=2)
        self.relu_optional = nn.ReLU()
        self.linear_patch_encoder = nn.Linear(embed_dim, patch_dim)

    def forward(self, audio_waveform):
        if audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.unsqueeze(1)
        audio_features = self.relu_optional(self.conv1d_optional(audio_waveform))
        audio_features = audio_features.transpose(1, 2)
        batch_size, seq_len, _ = audio_features.shape
        patches = []
        current_patch_features = []
        current_patch_amplitude_bytes = []
        for i in range(seq_len):
            feature_vector = audio_features[:, i:i+1, :]
            current_patch_features.append(feature_vector)
            discretized_feature_bytes = (feature_vector * 255).clamp(0, 255).round().int()
            current_patch_amplitude_bytes.append(discretized_feature_bytes)
            if current_patch_amplitude_bytes:
                current_patch_sequence_bytes = torch.cat(current_patch_amplitude_bytes, dim=1).squeeze(-1)
                with torch.no_grad():
                    next_byte_probs_tensor = self.entropy_predictor.get_next_byte_probs(current_patch_sequence_bytes)
                    entropy = calculate_shannon_entropy(next_byte_probs_tensor)
                if entropy.item() > self.entropy_threshold:
                    if current_patch_features:
                        patch_features_tensor = torch.cat(current_patch_features, dim=1)
                        encoded_patch = self.linear_patch_encoder(patch_features_tensor)
                        patches.append(encoded_patch)
                    current_patch_features = []
                    current_patch_amplitude_bytes = []
        if current_patch_features:
            patch_features_tensor = torch.cat(current_patch_features, dim=1)
            encoded_patch = self.linear_patch_encoder(patch_features_tensor)
            patches.append(encoded_patch)
        if patches:
            audio_patches_final = torch.cat(patches, dim=1)
        else:
            audio_patches_final = torch.zeros((batch_size, 0, self.patch_dim), dtype=torch.float32, device=audio_waveform.device)
        return audio_patches_final

# --- PDFEncoder for processing PDF modality ---
class PDFEncoder(nn.Module):
    def __init__(self, embed_dim, patch_dim, entropy_predictor, entropy_threshold=0.8):
        super(PDFEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.entropy_predictor = entropy_predictor
        self.entropy_threshold = entropy_threshold
        self.linear_patch_encoder = nn.Linear(embed_dim, patch_dim)
        self.byte_embedding = nn.Embedding(256, embed_dim)

    def forward(self, pdf_input):
        import io
        from llmpdfocr.app import extract_text
        combined_text = ""
        if isinstance(pdf_input, (list, tuple)):
            for file in pdf_input:
                if isinstance(file, bytes):
                    pdf_stream = io.BytesIO(file)
                else:
                    pdf_stream = file
                combined_text += extract_text(pdf_stream) + "\n"
        else:
            if isinstance(pdf_input, bytes):
                pdf_stream = io.BytesIO(pdf_input)
            else:
                pdf_stream = pdf_input
            combined_text = extract_text(pdf_stream)
        pdf_bytes = list(combined_text.encode("utf-8"))
        if not pdf_bytes:
            return torch.zeros((1, 0, self.patch_dim), dtype=torch.float32)
         # Enhanced dynamic patching: segment PDF bytes based on global and relative entropy.
        patches_bytes = entropy_patching_global_threshold(pdf_bytes, self.entropy_predictor, global_threshold=self.entropy_threshold, relative_threshold=0.1)
        embeddings_list = []
        for patch in patches_bytes:
            patch_tensor = torch.tensor([list(patch)], dtype=torch.long, device=self.byte_embedding.weight.device)
            embeddings = self.byte_embedding(patch_tensor)
            patch_embedding = embeddings.mean(dim=1)
            encoded_patch = self.linear_patch_encoder(patch_embedding)
            embeddings_list.append(encoded_patch)
        if embeddings_list:
            pdf_patches = torch.cat(embeddings_list, dim=1)
        else:
            pdf_patches = torch.zeros((1, 0, self.patch_dim), dtype=torch.float32)
        return pdf_patches

# --- Video Encoder Integration with Cross-Attention and Entropy Prediction ---
class VideoEncoder(nn.Module):
    def __init__(self, patch_size, embed_dim, video_entropy_predictor, entropy_threshold=0.8, num_heads=4):
        super(VideoEncoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.video_entropy_predictor = video_entropy_predictor
        self.entropy_threshold = entropy_threshold
        self.num_heads = num_heads
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=embed_dim,
                                 kernel_size=(3, patch_size, patch_size),
                                 stride=(2, patch_size, patch_size),
                                 padding=(1, 0, 0))
        self.relu = nn.ReLU()
        self.binary_proj = nn.Linear(embed_dim, embed_dim)
        # Cross-Attention module to group tokens dynamically
        self.cross_attention = DecoderCrossAttention(config=None, input_dim=embed_dim, byte_dim=embed_dim, num_heads=num_heads)
        self.group_query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, video_tensor):
        return self.encode_video(video_tensor)

    def encode_video(self, video_tensor):
        x = self.conv3d(video_tensor)
        x = self.relu(x)
        return self.apply_dynamic_binary_patch(x)

    def apply_dynamic_binary_patch(self, features):
        B, E, T, H_new, W_new = features.shape
        x = features.view(B, E, T * H_new * W_new).permute(0, 2, 1)  # [B, tokens, embed_dim]
        patches = []
        current_patch_tokens = []
        current_patch_bytes = []
        for i in range(x.shape[1]):
            token = x[:, i:i+1, :]  # [B, 1, embed_dim]
            current_patch_tokens.append(token)
            token_byte = (token.mean(dim=-1) * 255).clamp(0, 255).round().int()  # [B, 1]
            current_patch_bytes.append(token_byte)
            if current_patch_bytes:
                patch_byte_seq = torch.cat(current_patch_bytes, dim=1)  # [B, length]
                with torch.no_grad():
                    probs = self.video_entropy_predictor.get_next_byte_probs(patch_byte_seq)
                    entropy = calculate_shannon_entropy(probs)
                if entropy.item() > self.entropy_threshold:
                    patch_tensor = torch.cat(current_patch_tokens, dim=1)
                    query = self.group_query.expand(B, -1, -1)
                    grouped_patch = self.cross_attention(query, patch_tensor)
                    encoded_patch = self.binary_proj(grouped_patch)
                    patches.append(encoded_patch)
                    current_patch_tokens = []
                    current_patch_bytes = []
        if current_patch_tokens:
            patch_tensor = torch.cat(current_patch_tokens, dim=1)
            query = self.group_query.expand(B, -1, -1)
            grouped_patch = self.cross_attention(query, patch_tensor)
            encoded_patch = self.binary_proj(grouped_patch)
            patches.append(encoded_patch)
        if patches:
            video_patches = torch.cat(patches, dim=1)
        else:
            video_patches = torch.zeros(B, 0, self.embed_dim, device=features.device)
        return video_patches

class SONARtoBytePatch(nn.Module):
    """
    Projects SONAR embeddings into the binary latent transformer space.
    Uses a linear projection, optionally applying a SONAR encoder to compute embeddings.
    If an encoder is provided, it processes the input and averages over the sequence dimension.
    Otherwise, it assumes the input is already in embedding space.
    """
    def __init__(self, sonar_dim, patch_dim, encoder=None):
        super(SONARtoBytePatch, self).__init__()
        self.sonar_dim = sonar_dim
        self.patch_dim = patch_dim
        self.projection = nn.Linear(sonar_dim, patch_dim)
        self.encoder = encoder

    def forward(self, sonar_input):
        if self.encoder is not None:
            sonar_output = self.encoder(sonar_input)
            # Average over the sequence dimension of the encoded output.
            embeddings = sonar_output.encoded_seqs.mean(dim=1)
        else:
            embeddings = sonar_input
        return self.projection(embeddings)

# --- Switching Gate Attention Module ---
class SwitchingGateAttention(nn.Module):
    def __init__(self, patch_dim, num_modalities):
        super(SwitchingGateAttention, self).__init__()
        self.patch_dim = patch_dim
        self.num_modalities = num_modalities
        self.gate_linear = nn.Linear(patch_dim, num_modalities)

    def forward(self, x):
        # Compute gating weights for each modality based on input embeddings.
        # x can be of shape (batch, patch_dim) or (batch, num_patches, patch_dim).
        if x.dim() == 3:
            x = x.mean(dim=1)
        gating_logits = self.gate_linear(x)
        gating_weights = torch.softmax(gating_logits, dim=-1)
        return gating_weights

# --- MultiModalEncoder ---
class MultiModalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, sonar_dim, patch_dim, audio_entropy_predictor):
        super(MultiModalEncoder, self).__init__()
        self.audio_encoder = AudioEncoder(embed_dim, patch_dim, audio_entropy_predictor)
        self.text_encoder = LocalEncoder(...)  # Placeholder for text encoder
        # For video, use our enhanced VideoEncoder. We pass audio_entropy_predictor as video_entropy_predictor.
        self.video_encoder = VideoEncoder(patch_size=16, embed_dim=768, video_entropy_predictor=audio_entropy_predictor, entropy_threshold=0.8, num_heads=4)
        self.pdf_encoder = PDFEncoder(embed_dim, patch_dim, audio_entropy_predictor)
        self.sonar_projector = SONARtoBytePatch(sonar_dim, patch_dim)  # Placeholder for SONAR projector; now fully implemented. 
        self.modalities = {
            "audio": self.audio_encoder,
            "text": self.text_encoder,
            "video": self.video_encoder,
            "pdf": self.pdf_encoder,
            "sonar": self.sonar_projector
        }
        self.switch_gate = SwitchingGateAttention(patch_dim, num_modalities=len(self.modalities))

    def apply_switch_gate(self, file_encodings: dict):
        """
        Applies the switching gate attention mechanism to combine modality-specific encodings.
        Args:
            file_encodings (dict): A dictionary where the keys are modality names (e.g., "audio", "text", "video", "pdf", "sonar")
                                   and the values are encoded tensors with shape (batch, patch_dim) or (batch, num_patches, patch_dim).
        Returns:
            A tuple (combined, gate_weights) where:
                combined is a tensor of shape (batch, patch_dim) representing the weighted combination of modality encodings,
                gate_weights is a tensor of shape (batch, num_modalities) representing the gating weights.
        """
        modality_reps = []
        for modality in self.modalities:
            if modality in file_encodings:
                encoding = file_encodings[modality]
                if encoding.dim() == 3:
                    encoding = encoding.mean(dim=1)
                modality_reps.append(encoding)
            else:
                raise ValueError(f"Missing encoding for modality: {modality}")
        stacked = torch.stack(modality_reps, dim=1)  # shape: (batch, num_modalities, patch_dim)
        mean_rep = stacked.mean(dim=1)  # shape: (batch, patch_dim)
        gate_weights = self.switch_gate(mean_rep)  # shape: (batch, num_modalities)
        weighted = gate_weights.unsqueeze(-1) * stacked  # shape: (batch, num_modalities, patch_dim)
        combined = weighted.sum(dim=1)  # shape: (batch, patch_dim)
        return combined, gate_weights

# --- RelevancePredictor ---
class RelevancePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, latent_state):
        return self.predictor_net(latent_state)

# --- ByteEntropyPredictor ---
class ByteEntropyPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ff_dim):
        super().__init__()
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    def forward(self, byte_sequences):
        byte_embeddings = self.byte_embedding(byte_sequences)
        memory = torch.zeros_like(byte_embeddings)
        decoder_output = self.transformer_decoder(byte_embeddings, memory)
        next_byte_logits = self.fc_out(decoder_output)
        next_byte_probs = torch.softmax(next_byte_logits, dim=-1)
        return next_byte_probs
    def get_next_byte_probs(self, byte_sequence_segment):
        return self.forward(byte_sequence_segment)[:, -1, :]

# --- LocalEncoder ---
class LocalEncoder(nn.Module):
    def __init__(self, config, vocab_size, hidden_size, num_layers_enc, num_heads, ff_dim, window_size_enc, entropy_predictor):
        super().__init__(config)
        self.num_layers_enc = num_layers_enc
        self.hidden_size = hidden_size
        self.window_size_enc = window_size_enc
        self.entropy_predictor = entropy_predictor
        self.entropy_threshold = 0.8
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim,
                                        batch_first=True, activation='relu', norm_first=True)
            for _ in range(num_layers_enc)
        ])
    def forward(self, byte_sequences):
        batch_size, seq_len = byte_sequences.shape
        patches = []
        current_patch_bytes = []
        current_patch_representations = []
        for i in range(seq_len):
            byte_val = byte_sequences[:, i:i+1]
            current_patch_bytes.append(byte_val)
            byte_embedding = self.byte_embedding(byte_val)
            current_patch_representations.append(byte_embedding)
            if current_patch_bytes:
                current_patch_sequence = torch.cat(current_patch_bytes, dim=1)
                with torch.no_grad():
                    next_byte_probs_tensor = self.entropy_predictor.get_next_byte_probs(current_patch_sequence)
                    entropy = calculate_shannon_entropy(next_byte_probs_tensor)
                if entropy.item() > self.entropy_threshold:
                    if current_patch_representations:
                        patch_representation = torch.cat(current_patch_representations, dim=1)
                        encoded_patch = patch_representation
                        for encoder_layer in self.encoder_layers:
                            encoded_patch = encoder_layer(encoded_patch)
                        patches.append(encoded_patch)
                    current_patch_bytes = []
                    current_patch_representations = []
        if current_patch_representations:
            patch_representation = torch.cat(current_patch_representations, dim=1)
            encoded_patch = patch_representation
            for encoder_layer in self.encoder_layers:
                encoded_patch = encoder_layer(encoded_patch)
            patches.append(encoded_patch)
        if patches:
            patch_representations_final = torch.cat(patches, dim=1)
        else:
            patch_representations_final = torch.zeros((batch_size, 0, self.hidden_size), dtype=torch.float32, device=byte_sequences.device)
        return patch_representations_final

def calculate_shannon_entropy(next_byte_probs_tensor):
    probs = next_byte_probs_tensor.squeeze(0)
    log_probs = torch.log2(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs)
    return entropy

def entropy_patching_global_threshold(byte_sequence, main_model, global_threshold=0.8, relative_threshold=0.1):
    patches = []
    current_patch_bytes = []
    prev_entropy = None
    for i, byte_val in enumerate(byte_sequence):
        current_patch_bytes.append(byte_val)
        if byte_val == ord('\n'):
            if current_patch_bytes:
                patches.append(bytes(current_patch_bytes))
                current_patch_bytes = []
                prev_entropy = None
            continue
        input_tensor = torch.tensor([current_patch_bytes], dtype=torch.long).to(main_model.device)
        with torch.no_grad():
            next_probs = main_model.get_next_byte_probs(input_tensor)
            if next_probs is None:
                current_entropy = 0.0
            else:
                current_entropy = calculate_shannon_entropy(next_probs).item()
        if prev_entropy is None:
            prev_entropy = current_entropy
        if current_entropy > global_threshold or (current_entropy - prev_entropy > relative_threshold):
            patches.append(bytes(current_patch_bytes))
            current_patch_bytes = []
            prev_entropy = None
        else:
            prev_entropy = min(prev_entropy, current_entropy)
    if current_patch_bytes:
        patches.append(bytes(current_patch_bytes))
    return patches

# --- Binary Patch Components ---

class BinaryPatchingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=0.5, temperature=1.0):
        """
        input_dim: Dimension of the continuous latent states.
        hidden_dim: Hidden dimension for computing the binary decision.
        threshold: Cutoff probability for deciding a patch boundary.
        temperature: Temperature for relaxed binary decisions (if needed).
        """
        super(BinaryPatchingModule, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 1)
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, latent_states):
        # latent_states shape: (batch, seq_len, input_dim)
        x = F.relu(self.linear(latent_states))
        logits = self.out_linear(x)  # shape: (batch, seq_len, 1)
        probs = torch.sigmoid(logits)
        # Create binary mask: condition = 1 indicates a patch boundary.
        binary_mask = (probs > self.threshold).float()
        # Use a straight-through estimator for differentiability.
        binary_mask = binary_mask + (probs - probs.detach())
        return binary_mask, probs

class PatchAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, pooling='mean', eos_detection_threshold=0.7, eos_pattern_length=3):
        """
        Aggregates contiguous latent states into patches based on binary boundaries.
        input_dim: Input dimension of latent states.
        output_dim: Projected output dimension (often equal to input_dim).
        pooling: Pooling method to combine states ('mean' or 'max').
        eos_detection_threshold: Threshold for detecting EOS patterns in continuous thoughts.
        eos_pattern_length: Number of consecutive states needed to confirm an EOS pattern.
        """
        super(PatchAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, output_dim)
        
        # EOS detection parameters
        self.eos_detection_threshold = eos_detection_threshold
        self.eos_pattern_length = eos_pattern_length
        
        # EOS pattern detector - learns to recognize end of thought patterns
        self.eos_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, latent_states, binary_mask):
        # latent_states: (batch, seq_len, input_dim)
        # binary_mask: (batch, seq_len, 1), where 1 indicates a patch boundary.
        batch_size, seq_len, _ = latent_states.shape
        patch_list = []
        eos_indices = []  # Track indices where EOS markers are found
        
        # First, detect potential EOS markers using the dedicated detector
        with torch.no_grad():
            eos_scores = self.eos_detector(latent_states).squeeze(-1)  # (batch, seq_len)
        
        for b in range(batch_size):
            current_patch = []
            patches = []
            
            # Track consecutive high EOS scores for pattern detection
            consecutive_eos_count = 0
            last_eos_idx = -1
            
            for i in range(seq_len):
                current_patch.append(latent_states[b, i])
                
                # Advanced EOS detection using both the dedicated detector and pattern recognition
                current_eos_score = eos_scores[b, i].item()
                
                # Check for EOS pattern - consecutive high scores or significant pattern change
                if current_eos_score > self.eos_detection_threshold:
                    consecutive_eos_count += 1
                    if consecutive_eos_count >= self.eos_pattern_length:
                        # Found an EOS pattern
                        eos_indices.append((b, i))
                        last_eos_idx = i
                else:
                    # Reset consecutive count if score drops below threshold
                    consecutive_eos_count = 0
                
                # Additional detection: Check for significant state transition patterns
                if i >= 2:
                    # Calculate state transition metrics
                    prev_state = latent_states[b, i-1]
                    current_state = latent_states[b, i]
                    
                    # Cosine similarity between consecutive states
                    cos_sim = F.cosine_similarity(prev_state.unsqueeze(0), current_state.unsqueeze(0), dim=1).item()
                    
                    # Detect sharp transitions (low similarity) followed by stable states
                    if cos_sim < 0.3 and current_eos_score > 0.5 and i != last_eos_idx:
                        eos_indices.append((b, i))
                        last_eos_idx = i
                
                if binary_mask[b, i, 0] == 1:
                    patch_tensor = torch.stack(current_patch, dim=0)  # (p, input_dim)
                    if self.pooling == 'mean':
                        pooled = torch.mean(patch_tensor, dim=0)
                    elif self.pooling == 'max':
                        pooled, _ = torch.max(patch_tensor, dim=0)
                    else:
                        pooled = patch_tensor[0]
                    patches.append(pooled)
                    current_patch = []  # Start a new patch.
            
            if current_patch:
                patch_tensor = torch.stack(current_patch, dim=0)
                if self.pooling == 'mean':
                    pooled = torch.mean(patch_tensor, dim=0)
                elif self.pooling == 'max':
                    pooled, _ = torch.max(patch_tensor, dim=0)
                else:
                    pooled = patch_tensor[0]
                patches.append(pooled)
            
            if len(patches) == 0:
                patches.append(torch.zeros(self.input_dim, device=latent_states.device))
            
            patch_list.append(torch.stack(patches))
        
        # Limit patch sequences to MAX_N_LATENT for the batch.
        max_patches = MAX_N_LATENT
        limited_patches = []
        for p in patch_list:
            if p.shape[0] > max_patches:
                p = p[:max_patches]
            elif p.shape[0] < max_patches:
                pad = torch.zeros(max_patches - p.shape[0], self.input_dim, device=latent_states.device)
                p = torch.cat([p, pad], dim=0)
            limited_patches.append(p)
        
        patches_tensor = torch.stack(limited_patches, dim=0)  # (batch, MAX_N_LATENT, input_dim)
        patches_tensor = self.proj(patches_tensor)
        
        # Create eos_bounds tuple if EOS markers were found
        eos_bounds = None
        if eos_indices:
            # Find the first and last EOS marker
            first_eos = min(eos_indices, key=lambda x: x[1])
            last_eos = max(eos_indices, key=lambda x: x[1])
            
            # Add context window around the EOS bounds for better transition
            context_window = 2  # Number of tokens to include before/after the actual EOS
            start_bound = max(0, first_eos[1] - context_window)
            end_bound = min(seq_len - 1, last_eos[1] + context_window)
            
            eos_bounds = (start_bound, end_bound)
            
            # Log detection for debugging
            if hasattr(self, 'log_eos_detection') and self.log_eos_detection:
                print(f"EOS bounds detected: {eos_bounds}, scores at bounds: "
                      f"{eos_scores[first_eos[0], start_bound].item():.3f} to "
                      f"{eos_scores[last_eos[0], end_bound].item():.3f}")
        
        return patches_tensor, eos_bounds

"""
COCONUT with Binary Dynamic Patches – Integrated Version

This file extends the standard continuous thought (Coconut) architecture by introducing binary dynamic patches.
A BinaryPatchingModule computes patch boundaries over continuous latent states.
A PatchAggregator groups these states into patches.
These patch embeddings are processed by a latent transformer.
Finally, the existing local_encoder (which should already be defined in this file or imported)
translates the processed patches into token IDs for readable output.
"""

# --- Integrated Model using Existing local_encoder for Binary Patch-to-Text Translation ---

# --- Deep Sleep Training Functions Step 1 in Training ---

def CalculateDeepSleepReward(current_state, action, previous_state, previous_action, deep_sleep_params):
    """
    Calculates the deep sleep reward based on the current state, current action,
    the previous state and previous action using the provided hyperparameters.
    """
    target_attention = deep_sleep_params['target_attention']
    target_compute = deep_sleep_params['target_compute']
    lambda_attention = deep_sleep_params['lambda_attention']
    lambda_compute = deep_sleep_params['lambda_compute']
    lambda_smoothness = deep_sleep_params['lambda_smoothness']

    current_attention = current_state['attention']
    current_compute = current_state['compute']
    previous_action_delta_a = previous_action['delta_attention']  # action assumed delta-based

    reward = - (
        lambda_attention * (current_attention - target_attention)**2 +
        lambda_compute * (current_compute - target_compute)**2 +
        lambda_smoothness * (action['delta_attention'] - previous_action_delta_a)**2
    )
    return reward

def save_checkpoint(step_name, model=None):
    import os, json, torch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "model_save"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Save model checkpoint as a safetensor file if a model is provided; otherwise, save dummy data.
    checkpoint_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}.safetensors")
    if model is not None:
        # In practice, one would use a dedicated safetensors library.
        torch.save(model.state_dict(), checkpoint_filename)
    else:
        with open(checkpoint_filename, "w") as f:
            f.write("Checkpoint data for " + step_name)
    print("Checkpoint saved:", checkpoint_filename)
    
    # Create a config.json file with instructions for model inference and architecture details.
    config_data = {
        "checkpoint_file": checkpoint_filename,
        "instructions": "To set up the COCONUT byte latent class model for inference, load the state_dict from this checkpoint file into your model and use the provided configuration parameters.",
        "model_architecture": {
            "model_type": "CoconutBinaryLatentModel",
            "components": {
                "continuous_model": "Transformer-based continuous thought generator",
                "binary_patch_module": "Dynamic binary patching for latent states",
                "patch_aggregator": "Groups latent states into coherent patches",
                "latent_transformer": "Processes patch embeddings",
                "local_encoder": "Translates latent patches to token IDs"
            },
            "sleep_system": {
                "deep_sleep_params": {
                    "target_attention": 0.1,
                    "target_compute": 0.2,
                    "lambda_attention": 1.0,
                    "lambda_compute": 1.0,
                    "lambda_smoothness": 0.5
                },
                "awakening_params": {
                    "target_attention": 0.9,
                    "target_compute": 0.9,
                    "lambda_attention": 1.0,
                    "lambda_compute": 1.0,
                    "lambda_smoothness": 0.5
                }
            },
            "additional_features": {
                "dynamic_patching": "Entropy-based dynamic patching for efficient processing",
                "consciousness_control": "Adjustable consciousness levels for resource management",
                "emergency_override": "Emergency awakening capability for critical situations"
            }
        }
    }
    config_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}_config.json")
    with open(config_filename, "w") as cf:
        json.dump(config_data, cf, indent=4)
    print("Config file saved:", config_filename)

def play_sound(sound_file):
    import subprocess, platform
    try:
        if platform.system() == "Linux":
            subprocess.run(["aplay", sound_file])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", sound_file])
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
        print("Sound played:", sound_file)
    except Exception as e:
        print("Failed to play sound:", e)

def deep_sleep_training():
    print("Starting Deep Sleep Training Step")
    # Define hyperparameters for deep sleep training
    deep_sleep_params = {
        'target_attention': 0.1,
        'target_compute': 0.2,
        'lambda_attention': 1.0,
        'lambda_compute': 1.0,
        'lambda_smoothness': 0.5
    }
    # Dummy values for demonstration – in practice these would be retrieved from the model/environment.
    previous_state = {'attention': 0.5, 'compute': 0.5, 'metric': 0.0}
    current_state = {'attention': 0.2, 'compute': 0.3, 'metric': 0.0}
    previous_action = {'delta_attention': 0.05, 'delta_compute': 0.05, 'delta_metric': 0.0}
    current_action = {'delta_attention': 0.03, 'delta_compute': 0.02, 'delta_metric': 0.0}

    reward = CalculateDeepSleepReward(current_state, current_action, previous_state, previous_action, deep_sleep_params)
    print("Deep Sleep Reward calculated:", reward)

    # Simulate the training step:
    save_checkpoint("deep_sleep_step")
    # (… training operations would be performed here …)
    save_checkpoint("deep_sleep_step_checkpoint")

    play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")

    input("Deep sleep training step completed. Press Enter to continue...")

    return reward

# --- Sleep and Awakening System ---
class SleepAwakeningSystem:
    def __init__(self, model, deep_sleep_params=None, awakening_params=None):
        """
        Initialize the Sleep and Awakening System.
        
        Args:
            model: The LLM model to control
            deep_sleep_params: Parameters for deep sleep (optional)
            awakening_params: Parameters for awakening (optional)
        """
        self.model = model
        self.deep_sleep_params = deep_sleep_params or {
            'target_attention': 0.1,
            'target_compute': 0.2,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5
        }
        self.awakening_params = awakening_params or {
            'target_attention': 0.9,
            'target_compute': 0.9,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5,
            'emergency_reward': 10.0,
            'emergency_confirmation_threshold': 3
        }
        
        # State tracking
        self.current_state = {'attention': 0.9, 'compute': 0.9, 'metric': 0.0}
        self.previous_state = {'attention': 0.9, 'compute': 0.9, 'metric': 0.0}
        self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
        
        # Sleep/wake status
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.emergency_counter = 0
        
        # Q-learning parameters
        self.q_table = {}  # State-action value function
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # For epsilon-greedy action selection
        
        # Gating mechanism
        self.attention_gate = nn.Parameter(torch.ones(1))
        self.compute_gate = nn.Parameter(torch.ones(1))
        
        # Consciousness level control (0.0 to 1.0, where 1.0 is full consciousness)
        self.consciousness_level = 1.0
        self.consciousness_gate = nn.Parameter(torch.ones(1))
        
    def update_state(self, new_attention=None, new_compute=None, new_metric=None):
        """Update the current state with new values."""
        self.previous_state = self.current_state.copy()
        
        if new_attention is not None:
            self.current_state['attention'] = new_attention
        if new_compute is not None:
            self.current_state['compute'] = new_compute
        if new_metric is not None:
            self.current_state['metric'] = new_metric
            
    def choose_action(self, state, is_emergency=False):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state dictionary
            is_emergency: Whether this is an emergency situation
            
        Returns:
            action: Dictionary with delta values
        """
        if is_emergency:
            # Emergency action to quickly wake up
            return {
                'delta_attention': max(0.9 - state['attention'], 0), #Need to fix Eos Bound to be sure that Eos is detected in the contineous thoughts to know when the model is done thinking.
                'delta_compute': max(0.9 - state['compute'], 0),
                'delta_metric': 0.0
            }
            
        # Convert state to a hashable representation
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return {
                'delta_attention': np.random.uniform(0, 0.2),
                'delta_compute': np.random.uniform(0, 0.2),
                'delta_metric': np.random.uniform(0, 0.1)
            }
        else:
            # Exploit: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # If no actions have been tried yet, return a default action
            if not self.q_table[state_key]:
                return {
                    'delta_attention': 0.1,
                    'delta_compute': 0.1,
                    'delta_metric': 0.0
                }
            
            # Find the action with the highest Q-value
            best_action_key = max(self.q_table[state_key], key=lambda k: self.q_table[state_key][k])
            return self._key_to_action(best_action_key)
    
    def _state_to_key(self, state):
        """Convert a state dictionary to a hashable key."""
        return (round(state['attention'], 2), round(state['compute'], 2), round(state['metric'], 2))
    
    def _action_to_key(self, action):
        """Convert an action dictionary to a hashable key."""
        return (round(action['delta_attention'], 2), round(action['delta_compute'], 2), round(action['delta_metric'], 2))
    
    def _key_to_action(self, key):
        """Convert a hashable key back to an action dictionary."""
        return {
            'delta_attention': key[0],
            'delta_compute': key[1],
            'delta_metric': key[2]
        }
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state dictionary
            action: Action taken dictionary
            reward: Reward received
            next_state: Next state dictionary
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Find max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        self.q_table[state_key][action_key] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state_key][action_key]
        )
    
    def apply_gating_mechanism(self, attention_tensor, compute_tensor):
        """
        Apply the gating mechanism to control attention, compute, and consciousness.
        
        Args:
            attention_tensor: Tensor representing attention
            compute_tensor: Tensor representing compute resources
            
        Returns:
            gated_attention: Gated attention tensor
            gated_compute: Gated compute tensor
        """
        gated_attention = attention_tensor * self.attention_gate
        gated_compute = compute_tensor * self.compute_gate
        
        # Apply consciousness gating to both attention and compute
        gated_attention = gated_attention * self.consciousness_gate
        gated_compute = gated_compute * self.consciousness_gate
        
        return gated_attention, gated_compute
    
    def set_consciousness_level(self, level):
        """
        Manually set the consciousness level of the model.
        This is only allowed to be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        # Ensure level is within valid range
        level = max(0.01, min(1.0, level))  # Never go completely to zero to avoid complete shutdown
        
        # Set the consciousness level
        self.consciousness_level = level
        
        # Update the gates to reflect the new consciousness level
        self.update_gates()
        
        print(f"Consciousness level set to: {level:.2f}")
        return self.consciousness_level
    
    def update_gates(self):
        """Update the gating mechanism based on current sleep state and consciousness level."""
        if self.is_fully_shutdown:
            # Fully shut off
            self.attention_gate.data = torch.zeros_like(self.attention_gate)
            self.compute_gate.data = torch.zeros_like(self.compute_gate)
            self.consciousness_gate.data = torch.zeros_like(self.consciousness_gate)
        elif self.is_sleeping:
            # Reduced activity during sleep
            self.attention_gate.data = torch.tensor([self.current_state['attention']])
            self.compute_gate.data = torch.tensor([self.current_state['compute']])
            self.consciousness_gate.data = torch.tensor([min(self.current_state['attention'], self.consciousness_level)])
        else:
            # Awake but consciousness may be manually adjusted
            self.attention_gate.data = torch.ones_like(self.attention_gate)
            self.compute_gate.data = torch.ones_like(self.compute_gate)
            self.consciousness_gate.data = torch.tensor([self.consciousness_level])
    
    def check_emergency(self, emergency_signal=None):
        """
        Check if there's an emergency that requires immediate awakening.
        
        Args:
            emergency_signal: External emergency signal (optional)
            
        Returns:
            is_emergency: Boolean indicating if emergency override should be triggered
        """
        # Update emergency counter based on signal
        if emergency_signal:
            self.emergency_counter += 1
        else:
            self.emergency_counter = max(self.emergency_counter - 1, 0)
        
        # Check if emergency threshold is reached
        return self.emergency_counter >= self.awakening_params['emergency_confirmation_threshold']
    
    def enter_deep_sleep(self):
        """Initiate the deep sleep process."""
        print("Initiating deep sleep process...")
        self.is_sleeping = True
        self.is_fully_shutdown = False
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # Run deep sleep training loop
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate reward
                reward = self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                
                # Update gates
                self.update_gates()
                
                # Check if target sleep state is reached
                if (abs(self.current_state['attention'] - self.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.deep_sleep_params['target_compute']) < 0.05):
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Deep sleep training episode {episode}, current state: {self.current_state}")
        
        # Final update to fully shut down if needed
        if self.current_state['attention'] <= 0.1 and self.current_state['compute'] <= 0.1:
            self.is_fully_shutdown = True
            self.update_gates()
            print("LLM has entered full shutdown mode.")
        
        # Save checkpoint
        save_checkpoint("deep_sleep_final", self.model)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return self.current_state
    
    def awaken(self, emergency_override=False):
        """
        Awaken the model from sleep state.
        
        Args:
            emergency_override: Whether to use emergency override
            
        Returns:
            final_state: The final state after awakening
        """
        if not self.is_sleeping and not self.is_fully_shutdown:
            print("Model is already awake.")
            return self.current_state
        
        print(f"Initiating awakening process{' with emergency override' if emergency_override else ''}...")
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # If emergency override, immediately set to awake state
        if emergency_override:
            self.current_state = {
                'attention': self.awakening_params['target_attention'],
                'compute': self.awakening_params['target_compute'],
                'metric': 0.0
            }
            self.is_sleeping = False
            self.is_fully_shutdown = False
            self.update_gates()
            
            # Calculate and apply emergency reward for learning
            emergency_reward = self.awakening_params['emergency_reward']
            emergency_action = {
                'delta_attention': self.awakening_params['target_attention'] - initial_state['attention'],
                'delta_compute': self.awakening_params['target_compute'] - initial_state['compute'],
                'delta_metric': 0.0
            }
            self.update_q_value(initial_state, emergency_action, emergency_reward, self.current_state)
            
            print("Emergency awakening completed.")
            return self.current_state
        
        # Regular gradual awakening
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Check for emergency
                if self.check_emergency():
                    return self.awaken(emergency_override=True)
                
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                reward = -self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                
                # Update gates
                self.update_gates()
                
                # Check if target awake state is reached
                if (abs(self.current_state['attention'] - self.awakening_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.awakening_params['target_compute']) < 0.05):
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Awakening training episode {episode}, current state: {self.current_state}")
        
        # Final update
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.update_gates()
        
        # Save checkpoint
        save_checkpoint("awakening_final", self.model)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print("Awakening process completed.")
        return self.current_state
    
    def calculate_deep_sleep_reward(self, current_state, action, previous_state, previous_action):
        """
        Calculate the deep sleep reward based on current and previous states and actions.
        
        Args:
            current_state: Current state dictionary
            action: Current action dictionary
            previous_state: Previous state dictionary
            previous_action: Previous action dictionary
            
        Returns:
            reward: Deep sleep reward
        """
        return CalculateDeepSleepReward(
            current_state, action, previous_state, previous_action, self.deep_sleep_params
        )

class Value(nn.Module):
    """Value function for the RL algorithm."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state):
        return self.value_net(state)

class CoconutBinaryLatentModel(nn.Module):
    def __init__(self, continuous_model, latent_transformer,  MultiModalEncoder, input_dim, hidden_dim, initial_temperature: float = 1.0, surprise_threshold: float = 0.5, ):
        """
        continuous_model: Module that outputs continuous latent representations (continuous thought).
        latent_transformer: Module (e.g., from Binary Latent Transformer) that processes patch embeddings.
        local_encoder: The existing local_encoder for converting latent patch outputs into token IDs (readable text).
        input_dim: Dimension of the continuous latent states.
        hidden_dim: Hidden dimension for the binary patch module.
        """
        super(CoconutBinaryLatentModel, self).__init__()
        self.continuous_model = continuous_model
        self.binary_patch_module = BinaryPatchingModule(input_dim, hidden_dim)
        self.patch_aggregator = PatchAggregator(input_dim, input_dim)
        self.latent_transformer = latent_transformer
        self.local_encoder =  MultiModalEncoder # Reuse the local encoder for final text translation.

        self.multi_encoder = MultiModalEncoder(
            vocab_size=256, 
            embed_dim=input_dim, 
            sonar_dim=512, 
            patch_dim=input_dim,
            audio_entropy_predictor=self.latent_transformer.entropy_predictor
        )
        self.local_decoder = LocalDecoder(
            config, 
            input_dim=input_dim, 
            output_bytes_dim=256, 
            num_layers=3, 
            num_heads=4, 
            ff_dim=128
        )

        self.surprise_threshold = surprise_threshold
        
        # Initialize sleep and awakening system
        self.sleep_system = SleepAwakeningSystem(self)

    def forget_memories(self, hours_ago=24, agent_info_id=None):
        import datetime
        time = 0 #Need to grab time from the current system date time to save to the memory layer with the memory so that the time the memory occured is saved with the corresponding memory.
        end_time = time.time()
        start_time = end_time - (hours_ago * 3600)
        self.memory_layer.forget_memories(start_time=start_time, end_time=end_time, agent_info_id=agent_info_id)

    def _get_marker_embedding(self, marker_text, embed_dim, device):
        """
        Create an embedding for a marker text (like "<output>" or "/<output>").
        
        Args:
            marker_text: The text of the marker
            embed_dim: Embedding dimension
            device: Device to create the tensor on
            
        Returns:
            A tensor of shape (1, 1, embed_dim) representing the marker embedding
        """
        # Simple hash-based embedding for demonstration
        marker_bytes = marker_text.encode('utf-8')
        marker_hash = sum(marker_bytes) % 10000
        
        # Use the hash to seed a random generator for reproducibility
        import numpy as np
        rng = np.random.RandomState(marker_hash)
        
        # Generate a random embedding vector
        embedding = torch.tensor(rng.normal(0, 0.02, embed_dim), dtype=torch.float32, device=device)
        
        # Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=0)
        
        # Reshape to (1, 1, embed_dim)
        return embedding.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        # Check if model is in full shutdown mode
        if hasattr(self, 'sleep_system') and self.sleep_system.is_fully_shutdown:
            # Return empty output if fully shut down
            batch_size = x.size(0)
            dummy_output = torch.zeros((batch_size, 1, 256), device=x.device)
            return dummy_output, None
        
        # Step 1: Generate continuous latent representations.
        latent_states = self.continuous_model(x)  # (batch, seq_len, input_dim)
        
        # Apply gating mechanisms (sleep and consciousness)
        if hasattr(self, 'sleep_system'):
            # Apply gating to latent states
            attention_tensor = latent_states  # Assuming this represents attention
            compute_tensor = latent_states    # Assuming this also affects compute resources
            gated_attention, gated_compute = self.sleep_system.apply_gating_mechanism(
                attention_tensor, compute_tensor
            )
            latent_states = gated_compute  # Use gated compute as the modified latent states
        
        # Step 2: Compute binary patch boundary decisions.
        binary_mask, probs = self.binary_patch_module(latent_states)
        
        # Step 3: Aggregate latent states into patches.
        patch_embeddings, eos_bounds = self.patch_aggregator(latent_states, binary_mask)
        
        # Continuous thought markers insertion has been delegated to patch_aggregator via reward training.
        
        # Step 4: Process the patches with the latent transformer.
        latent_output = self.latent_transformer(patch_embeddings)
        
        # If eos_bounds indicate the end of latent thinking, truncate latent_output accordingly.
        if eos_bounds is not None and isinstance(eos_bounds, tuple):
            latent_output = latent_output[:, eos_bounds[1]:, :]
        
        # Append final answer marker dynamically if output boundaries are provided.
        if hasattr(self.latent_transformer, 'output_bounds') and self.latent_transformer.output_bounds is not None and isinstance(self.latent_transformer.output_bounds, tuple):
            output_start, output_end = self.latent_transformer.output_bounds
            # Use the dynamically determined output boundaries to set the final output segment.
            latent_output = latent_output[:, output_start:output_end, :]
        else:
            final_marker = self._get_marker_embedding("<output>", latent_output.size(2), latent_output.device)
            final_end_marker = self._get_marker_embedding("/<output>", latent_output.size(2), latent_output.device)
            latent_output = torch.cat([latent_output, final_marker, final_end_marker], dim=1)
        
        # Step 5: Use the multi-encoder to translate latent patches into a unified encoded representation.
        # This allows the model to read the input file regardless of its modality.
        encoded_output = self.multi_encoder(latent_output)
        # Then, use the local patch decoder to translate the encoded output into a readable text format.
        # Create a dummy input sequence for the decoder.
        dummy_input = torch.randint(0, 256, (encoded_output.size(0), encoded_output.size(1)), dtype=torch.long, device=encoded_output.device)
        outputbinary = self.local_decoder(encoded_output, dummy_input)
        return outputbinary, eos_bounds
    

    def train(self):
        self.base_causallm.train()
    def eval(self):
        self.base_causallm.eval()
        
    def sleep(self):
        """
        Manually put the model to sleep (graceful shutdown).
        This can only be triggered by the user, not by the model itself.
        
        Returns:
            sleep_state: The final state after entering deep sleep
        """
        if hasattr(self, 'sleep_system'):
            print("User initiated sleep mode...")
            return self.sleep_system.enter_deep_sleep()
        else:
            print("Sleep system not initialized.")
            return None
    
    def wake_up(self, emergency=False):
        """
        Manually wake up the model if it's in sleep mode.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            emergency: Whether to use emergency override for immediate awakening
            
        Returns:
            awake_state: The final state after awakening
        """
        if hasattr(self, 'sleep_system'):
            if self.sleep_system.is_sleeping or self.sleep_system.is_fully_shutdown:
                print("User initiated wake up sequence...")
                return self.sleep_system.awaken(emergency_override=emergency)
            else:
                print("Model is already awake.")
                return self.sleep_system.current_state
        else:
            print("Sleep system not initialized.")
            return None
    
    def set_consciousness(self, level):
        """
        Manually set the consciousness level of the model.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        if hasattr(self, 'sleep_system'):
            print("User adjusting consciousness level...")
            return self.sleep_system.set_consciousness_level(level)
        else:
            print("Sleep system not initialized.")
            return None
    
    def train_sleep_wake_mechanisms(self, num_episodes=100, steps_per_episode=20):
        """
        Explicitly train the sleep and wake mechanisms using reinforcement learning.
        This allows the model to learn optimal policies for transitioning between
        sleep and wake states before actually using these mechanisms in production.
        
        Args:
            num_episodes: Number of training episodes for each mechanism
            steps_per_episode: Number of steps per episode
            
        Returns:
            training_results: Dictionary containing training metrics
        """
        if not hasattr(self, 'sleep_system'):
            print("Sleep system not initialized.")
            return None
            
        print("Starting training for sleep and wake mechanisms...")
        results = {
            'sleep': {'initial_state': None, 'final_state': None, 'episodes': []},
            'wake': {'initial_state': None, 'final_state': None, 'episodes': []}
        }
        
        # Train deep sleep mechanism
        print("\n=== Training Deep Sleep Mechanism ===")
        
        # Store initial state
        initial_sleep_state = self.sleep_system.current_state.copy()
        results['sleep']['initial_state'] = initial_sleep_state
        
        # Temporarily set to awake state for training
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        
        # Save initial checkpoint before training begins
        save_checkpoint("sleep_wake_training_start", self)
        
        # Track training progress for mid-training checkpoint
        mid_training_point = num_episodes // 2
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_sleep_state.copy()
            self.sleep_system.previous_state = initial_sleep_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for sleep, we decrease attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate reward
                reward = self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target sleep state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.deep_sleep_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['sleep']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Sleep training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['sleep']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint
            if episode == mid_training_point:
                save_checkpoint("sleep_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final sleep state
        results['sleep']['final_state'] = self.sleep_system.current_state.copy()
        
        # Save checkpoint after sleep mechanism training
        save_checkpoint("sleep_mechanism_complete", self)
        
        # Train awakening mechanism
        print("\n=== Training Awakening Mechanism ===")
        
        # Store initial state (low attention/compute)
        initial_wake_state = {
            'attention': self.sleep_system.deep_sleep_params['target_attention'],
            'compute': self.sleep_system.deep_sleep_params['target_compute'],
            'metric': 0.0
        }
        results['wake']['initial_state'] = initial_wake_state
        
        # Temporarily set to sleep state for training
        self.sleep_system.is_sleeping = True
        self.sleep_system.is_fully_shutdown = False
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_wake_state.copy()
            self.sleep_system.previous_state = initial_wake_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for wake, we increase attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                reward = -self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target awake state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.awakening_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.awakening_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['wake']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Wake training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['wake']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint for wake mechanism
            if episode == mid_training_point:
                save_checkpoint("wake_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final wake state
        results['wake']['final_state'] = self.sleep_system.current_state.copy()
        
        # Reset to normal awake state
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        self.sleep_system.current_state = {
            'attention': self.sleep_system.awakening_params['target_attention'],
            'compute': self.sleep_system.awakening_params['target_compute'],
            'metric': 0.0
        }
        self.sleep_system.update_gates()
        
        print("\nTraining completed. Sleep and wake mechanisms have been trained.")
        
        # Save final checkpoint
        save_checkpoint("sleep_wake_training_complete", self)
        
        # Play sound to indicate training completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return results


class LocalDecoder(nn.Module):
    def __init__(self, config, input_dim, output_bytes_dim, num_layers, num_heads, ff_dim, byte_dim=None):
        super().__init__(config)
        self.num_layers = num_layers
        self.output_bytes_dim = output_bytes_dim
        self.byte_dim = byte_dim if byte_dim is not None else output_bytes_dim
        self.byte_embedding = nn.Embedding(output_bytes_dim, self.byte_dim)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, input_dim=input_dim, byte_dim=self.byte_dim, num_heads=num_heads, ff_dim=ff_dim)
                                              for _ in range(num_layers)])
        self.final_linear = nn.Linear(self.byte_dim, output_bytes_dim)
    def forward(self, patch_representations, byte_sequence_input):
        byte_representations = self.byte_embedding(byte_sequence_input)
        for decoder_layer in self.decoder_layers:
            byte_representations = decoder_layer(patch_representations, byte_representations)
        return self.final_linear(byte_representations)

class DecoderLayer(nn.Module):
    def __init__(self, config, input_dim, byte_dim, num_heads, ff_dim):
        super().__init__(config)
        self.cross_attention = DecoderCrossAttention(config, input_dim=input_dim, byte_dim=byte_dim, num_heads=num_heads)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=byte_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
    def forward(self, patch_representations, byte_representations):
        cross_attn_output = self.cross_attention(patch_representations, byte_representations)
        return self.transformer_layer(cross_attn_output, memory=None)

class DecoderCrossAttention(nn.Module):
    def __init__(self, config, input_dim, byte_dim, num_heads):
        super().__init__(config)
        self.cross_attn = nn.MultiheadAttention(embed_dim=byte_dim, num_heads=num_heads, batch_first=True)
        self.wq = nn.Linear(byte_dim, byte_dim)
        self.wk = nn.Linear(input_dim, byte_dim)
        self.wv = nn.Linear(input_dim, byte_dim)
        self.dense = nn.Linear(byte_dim, byte_dim)
        self.norm_q = nn.LayerNorm(byte_dim)
        self.norm_k = nn.LayerNorm(input_dim)
        self.norm_v = nn.LayerNorm(input_dim)
    def forward(self, patch_representations, byte_representations):
        query = self.norm_q(self.wq(byte_representations))
        key = self.norm_k(self.wk(patch_representations))
        value = self.norm_v(self.wv(patch_representations))
        attn_output, _ = self.cross_attn(query, key, value)
        output = self.dense(attn_output)
        return output + byte_representations

# --- Fallback Video Encoder (Simple) ---
class VideoEncoderSimple(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(VideoEncoderSimple, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=embed_dim,
                                kernel_size=(3, patch_size, patch_size),
                                stride=(2, patch_size, patch_size),
                                padding=(1, 0, 0))
        self.relu = nn.ReLU()
        self.binary_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, video_tensor):
        x = self.conv3d(video_tensor)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        binary_tokens = torch.sigmoid(self.binary_proj(x))
        return (binary_tokens > 0.5).float()


# --- End of Deep Sleep Training Functions ---

if __name__ == '__main__':
    config = namedtuple("Config", [])()
    
    # Create a dummy continuous model for testing
    continuous_model = nn.Sequential(
        nn.Linear(256, 64),
        nn.ReLU()
    )
    
    # Create the CoconutBinaryLatentModel
    coconut_model = CoconutBinaryLatentModel(
        continuous_model=continuous_model,
        latent_transformer=CoconutBinaryLatentModel,
        local_encoder=CoconutBinaryLatentModel.multiencoder,
        input_dim=64,
        hidden_dim=32
    )
    
    # Test input
    input_byte_sequence = b"<eos>What is the capital of France?/<eos><output>The capital is Paris.</output>"
    input_ids_example = torch.tensor([[byte for byte in input_byte_sequence]], dtype=torch.long)
    
    # Process the input through the CoconutBinaryLatentModel
    print("Processing input through CoconutBinaryLatentModel...")
    output_binary, eos_bounds = coconut_model(input_ids_example)
    print(f"Output shape: {output_binary.shape}, EOS bounds: {eos_bounds}")
    
    # Demonstrate sleep functionality
    print("\nDemonstrating deep sleep mode...")
    sleep_state = coconut_model.sleep_system.enter_deep_sleep()
    print(f"Sleep state: {sleep_state}")
    
    # Demonstrate emergency awakening
    print("\nDemonstrating emergency awakening...")
    awake_state = coconut_model.sleep_system.awaken(emergency_override=True)
    print(f"Awake state: {awake_state}")
