import torch
import torch.nn as nn
import datetime
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
from NueralMemoryLayers import HierarchicalMemory, MemoryNode
from OldCOCONUTUnused.StableCELoss import stable_cross_entropy_loss
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss

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
    def __init__(self, input_dim, output_dim, pooling='mean'):
        """
        Aggregates contiguous latent states into patches based on binary boundaries.
        input_dim: Input dimension of latent states.
        output_dim: Projected output dimension (often equal to input_dim).
        pooling: Pooling method to combine states ('mean' or 'max').
        """
        super(PatchAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, latent_states, binary_mask):
        # latent_states: (batch, seq_len, input_dim)
        # binary_mask: (batch, seq_len, 1), where 1 indicates a patch boundary.
        batch_size, seq_len, _ = latent_states.shape
        patch_list = []
        for b in range(batch_size):
            current_patch = []
            patches = []
            for i in range(seq_len):
                current_patch.append(latent_states[b, i])
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
        return patches_tensor

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

class CoconutBinaryLatentModel(nn.Module):
    def __init__(self, continuous_model, latent_transformer, local_encoder, input_dim, hidden_dim):
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
        self.local_encoder = local_encoder   # Reuse the local encoder for final text translation.
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

    def forward(self, x):
        # Step 1: Generate continuous latent representations.
        latent_states = self.continuous_model(x)  # (batch, seq_len, input_dim)
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

# --- BinaryLatentTransformer ---
class BinaryLatentTransformer(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, ff_dim: int, sensory_input_channels: int, config,
                 max_states: Optional[int] = None, patch_size: int = 4, num_latent_states: int = 4,
                 reflection_threshold: float = 0.5, state_history_size=5, initial_temperature: float = 1.0,
                 b_star_n_star: int = 4,
                 memory_layer: Optional[HierarchicalMemory] = None, surprise_threshold: float = 0.5,
                 memory_influence_factor: float = 0.5, state_quality_threshold: float = 0.5,
                 replacement_rate: float = 1e-4, decay_rate: float = 0.99, maturity_threshold: int = 100,
                 vocab_size: int = 256, num_layers_enc: int = 2, window_size_enc: int = 3, num_layers_entropy_pred: int = 1):
        super().__init__(config)
        self.memory_layer = memory_layer if memory_layer is not None else HierarchicalMemory(
            num_layers=4, root_memory_chunk_size=(hidden_size,), cache_capacity=10000
        ).requires_grad_(False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_states = max_states
        self.patch_size = patch_size
        self.num_latent_states = num_latent_states
        self.reflection_threshold = reflection_threshold
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.patch_encoder = nn.Linear(self.patch_size * hidden_size, hidden_size)
        self.compression_enabled = max_states is not None
        if self.compression_enabled:
            self.tova_query = nn.Linear(hidden_size, hidden_size)
            self.tova_key = nn.Linear(hidden_size, hidden_size)
            self.tova_value = nn.Linear(hidden_size, hidden_size)
        self.latent_mode = True
        self.thought_conditioning = True
        self.state_selector = nn.Linear(hidden_size, 1)
        self.state_history_size = state_history_size
        self.state_evaluator_fc = nn.Linear(hidden_size, 1)
        self.reward_generator = nn.Linear(hidden_size, 1)
        self.state_history_buffer = []
        self.temperature = initial_temperature
        self.b_star_n_star = b_star_n_star
        self.current_step = 0
        self.adaptation_interval = 500
        self.evaluation_set_size = 600
        self.exploration_batch_size = 10
        self.exploration_batch = []
        self.units_to_replace_count = {f'layer{i}': 0 for i in range(num_layers)}
        self.unit_ages = {f'layer{i}': torch.zeros(num_heads, dtype=torch.int) for i in range(num_layers)}
        self.prefrontal_cortex = PrefrontalCortex(hidden_size, num_layers, binary_latent_transformer=self)
        self.value_function = Value(hidden_size)
        self.surprise_threshold = surprise_threshold
        self.memory_influence_factor = memory_influence_factor
        self.state_quality_threshold = state_quality_threshold
        self.exploration_data = None
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.units_to_replace_count = {f'layer{i}': 0 for i in range(num_layers)}
        self.unit_ages = {f'layer{i}': torch.zeros(num_heads, dtype=torch.int) for i in range(num_layers)}
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                                     dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.local_decoder = LocalDecoder(config, input_dim=hidden_size, output_bytes_dim=vocab_size,
                                          num_layers=3, num_heads=num_heads, ff_dim=ff_dim)
        self.quality_predictor = nn.Linear(hidden_size, vocab_size)
        
        # Initialize Audio ByteEntropyPredictor
        audio_vocab_size = 256
        audio_entropy_predictor = ByteEntropyPredictor(
            vocab_size=audio_vocab_size, hidden_size=64, num_layers=1, num_heads=2, ff_dim=128
        )
        multimodal_encoder = MultiModalEncoder(
            vocab_size=256, embed_dim=256, sonar_dim=512, patch_dim=256, audio_entropy_predictor=audio_entropy_predictor
        )
        self.entropy_predictor = ByteEntropyPredictor(
            vocab_size=vocab_size, hidden_size=hidden_size // 2, num_layers=num_layers_entropy_pred,
            num_heads=num_heads // 2, ff_dim=ff_dim // 2
        )
        self.local_encoder = LocalEncoder(
            config, vocab_size=vocab_size, hidden_size=hidden_size, num_layers_enc=num_layers_enc,
            num_heads=num_heads, ff_dim=ff_dim, window_size_enc=window_size_enc, entropy_predictor=self.entropy_predictor
        )

    def transformer_encoder(self, src, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None,
                              past_key_values: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False):
        if past_key_values is None or not use_cache:
            output = self.transformer.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
            past_key_values = None
        else:
            output = self.transformer.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask,
                                              past_key_values=past_key_values, use_cache=use_cache)
        TransformerEncoderLayerOutput = namedtuple("TransformerEncoderLayerOutput", ["last_hidden_state", "past_key_values"])
        return TransformerEncoderLayerOutput(last_hidden_state=output, past_key_values=past_key_values)


    def get_next_byte_probs(self, byte_sequence_segment):
        try:
            patch_representations_entropy = self.local_encoder(byte_sequence_segment)
            encoder_output_entropy_output = self.transformer_encoder(patch_representations_entropy)
            encoder_output_entropy = encoder_output_entropy_output.last_hidden_state
            batch_size, seq_len, _ = encoder_output_entropy.shape
            byte_sequence_input_entropy = torch.zeros((batch_size, seq_len), dtype=torch.long,
                                                      device=encoder_output_entropy.device)
            decoded_bytes_output_tensor = self.local_decoder(encoder_output_entropy, byte_sequence_input_entropy)
            next_byte_probs = torch.softmax(decoded_bytes_output_tensor[:, -1, :], dim=-1)
            return next_byte_probs
        except Exception as e:
            print(f"Error during entropy prediction: {e}")
            return None

    def generate_binary_patches(self, input_ids, attention_mask, max_new_patches=16, output_embedding=False,
                                  synced_gpus=False, eos_threshold=0.9, **kwargs):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        input_ids_gen = input_ids.clone()
        generated_patches = []
        patch_sizes_history = []
        generated_byte_sequences = []
        latent_thinking_mode = False
        output_generation_mode = False
        next_compute_range = (0, input_ids_gen.shape[1])
        inputs_embeds = input_ids_gen
        kv_cache = None
        current_input_byte_sequence = b"".join([bytes([byte_id]) for byte_id in input_ids_gen[0].tolist()])
        if self.eos_think_byte_sequence in current_input_byte_sequence:
            print("Start of thinking phase <THINK_EOS> detected in input.")
            latent_thinking_mode = True
        for pass_idx in range(max_new_patches + MAX_N_LATENT):
            patch_representations = self.local_encoder(inputs_embeds[:, next_compute_range[0]:next_compute_range[1]])
            outputs = self.transformer_encoder(patch_representations, past_key_values=kv_cache, use_cache=True)
            latent_output = outputs.last_hidden_state
            kv_cache = outputs.past_key_values
            next_compute_range = (next_compute_range[1], input_ids_gen.shape[1])
            inputs_embeds = inputs_embeds
            if pass_idx >= MAX_N_LATENT and inputs_embeds.shape[1] < input_ids_gen.shape[1] + max_new_patches:
                binary_patch_output = latent_output[0, -1, :]
                generated_patches.append(binary_patch_output.detach().cpu())
                byte_embeddings_init_generate = torch.zeros_like(latent_output)
                batch_size_gen, seq_len_gen, _ = latent_output.shape
                byte_sequence_input_generate = torch.zeros((batch_size_gen, seq_len_gen), dtype=latent_output.device)
                decoded_bytes_output_tensor = self.local_decoder(latent_output, byte_sequence_input_generate)
                generated_byte_sequence = self.binary_patch_to_bytes(decoded_bytes_output_tensor[0, -1, :])
                generated_byte_sequences.append(generated_byte_sequence)
                accumulated_byte_sequence = b"".join(generated_byte_sequences)
                if latent_thinking_mode and self.eos_think_end_byte_sequence in accumulated_byte_sequence:
                    print("End of thinking phase </THINK_EOS> detected in generated output.")
                    latent_thinking_mode = False
                    output_generation_mode = True
                if not output_generation_mode and self.eos_final_output_start_byte_sequence in accumulated_byte_sequence:
                    print("Start of output generation <output> tag detected.")
                    output_generation_mode = True
                patch_size = len(generated_byte_sequence)
                patch_sizes_history.append(patch_size)
                print(f"Generated patch size: {patch_size} bytes")
                if output_generation_mode and accumulated_byte_sequence.endswith(self.eos_final_output_end_byte_sequence):
                    print("End of final output generation </output> tag detected, stopping generation.")
                    break
                if output_generation_mode:
                    current_byte_sequence_for_patching = b"".join(generated_byte_sequences)[-512:]
                    if current_byte_sequence_for_patching:
                        dynamic_patches = entropy_patching_global_threshold(current_byte_sequence_for_patching, self.base_causallm)
                        print(f"Dynamic patches generated: {len(dynamic_patches)}")
                new_patch_embed = binary_patch_output.unsqueeze(0).unsqueeze(0)
                inputs_embeds = torch.cat((inputs_embeds, new_patch_embed), dim=1)
        if synced_gpus:
            while self.gen_forward_cnt < max_new_patches + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.transformer_encoder(inputs_embeds)
        final_byte_sequence = b"".join(generated_byte_sequences)
        start_tag = self.eos_final_output_start_byte_sequence
        end_tag = self.eos_final_output_end_byte_sequence
        start_index = final_byte_sequence.find(start_tag)
        end_index = final_byte_sequence.rfind(end_tag)
        if start_index != -1 and end_index != -1 and start_index < end_index:
            user_output_bytes = final_byte_sequence[start_index + len(start_tag):end_index]
        else:
            user_output_bytes = final_byte_sequence
        if output_embedding:
            return generated_patches, user_output_bytes
        else:
            return user_output_bytes

    def binary_patch_to_bytes(self, decoded_bytes_output_tensor):
        predicted_byte_index = torch.argmax(decoded_bytes_output_tensor).item()
        generated_byte = bytes([predicted_byte_index])
        return generated_byte

    def forget_memories(self, hours_ago=24, agent_info_id=None):
        import datetime
        time = 0 #Need to grab time from the current system date time to save to the memory layer with the memory so that the time the memory occured is saved with the corresponding memory.
        end_time = time.time()
        start_time = end_time - (hours_ago * 3600)
        self.memory_layer.forget_memories(start_time=start_time, end_time=end_time, agent_info_id=agent_info_id)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

class PrefrontalCortex(nn.Module):
    def __init__(self, hidden_size, num_layers=3, binary_latent_transformer=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.binary_latent_transformer = binary_latent_transformer
        self.metacognitive = MetacognitiveModule(hidden_size, hidden_size)
    def forward(self, continuous_thought):
        latent_states = continuous_thought
        safety_report = self.metacognitive(latent_states, latent_states)
        if safety_report['needs_reflection']:
            corrected_states = safety_report['corrected_state']
            self.binary_latent_transformer.memory_layer.store(corrected_states.detach(), tags=['safety_correction_latent'])
            return corrected_states
        return latent_states

class MetacognitiveModule(nn.Module):
    def __init__(self, hidden_dim, memory_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.thought_monitor = nn.Linear(hidden_dim, 1)
        self.memory_monitor = nn.Linear(memory_dim, 1)
        self.reflection_net = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.error_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.reflection_memory = []
        self.max_reflections = 5

    def forward(self, continuous_thought, memory):
        thought_score = torch.sigmoid(self.thought_monitor(continuous_thought))
        memory_score = torch.sigmoid(self.memory_monitor(memory))
        safety_flag = (thought_score + memory_score) / 2
        combined = torch.cat([continuous_thought, memory], dim=-1)
        reflection = self.reflection_net(combined)
        error_prob = self.error_detector(reflection)
        if len(self.reflection_memory) >= self.max_reflections:
            self.reflection_memory.pop(0)
        self.reflection_memory.append(reflection.detach())
        corrected_state = continuous_thought
        if error_prob > 0.5:
            correction_input = torch.cat([continuous_thought, reflection], dim=-1)
            corrected_state = self.correction_net(correction_input)
        return {'safety_flag': safety_flag, 'reflection': reflection, 'error_prob': error_prob,
                'corrected_state': corrected_state, 'needs_reflection': error_prob > 0.5}


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


# --- Deep Sleep Training Functions ---

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
    checkpoint_dir = "checkpointLLMSaves"
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
    
    # Create a config.json file with instructions for model inference.
    config_data = {
        "checkpoint_file": checkpoint_filename,
        "instructions": "To set up the COCONUT byte latent class model for inference, load the state_dict from this checkpoint file into your model and use the provided configuration parameters."
    }
    config_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}_config.json")
    with open(config_filename, "w") as cf:
        json.dump(config_data, cf, indent=4)
    print("Config file saved:", config_filename)

def play_sound(sound_file):
    import subprocess
    try:
        subprocess.run(["aplay", sound_file])
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

# --- End of Deep Sleep Training Functions ---

if __name__ == '__main__':
    config = namedtuple("Config", [])()
    blt_model = BinaryLatentTransformer(config=config, hidden_size=64, num_layers=2,
                                        num_heads=4, ff_dim=128, sensory_input_channels=3, vocab_size=256)
    coconut_model = Coconut(base_causallm=blt_model)
    input_byte_sequence = b"<THINK_EOS>What is the capital of France?</THINK_EOS><output>The capital is Paris.</output>"
    input_ids_example = torch.tensor([[byte for byte in input_byte_sequence]], dtype=torch.long)
    generated_output_bytes = coconut_model.generate_binary_patches(input_ids_example, None)
    try:
        decoded_output = generated_output_bytes.decode('utf-8')
        print(f"Generated Output: {decoded_output}")
    except UnicodeDecodeError:
        print(f"Generated Output (raw bytes): {generated_output_bytes}")
