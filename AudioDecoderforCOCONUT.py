#Audio Decoder that needs to be placed into the COCONUTWLatentThinking.py file when training is finished. 
# Need to place by the other decoders. Decoder needs to decode both audio and text so both decoders need to work at once. 

import torch
import torch.nn as nn
from TOVACompression import TOVACompression

# --- AudioDecoder for generating audio from latent representations ---
class AudioDecoder(nn.Module):
    """
    AudioDecoder based on CosyVoice 2's flow matching approach.
    Converts latent representations back into audio waveforms using
    conditional flow matching with binary dynamic patches.
    
    Features TOVA compression for efficient memory usage during long audio generation.
    """
    def __init__(self, input_dim, hidden_dim, mel_dim=80, sample_rate=24000, 
                 num_flow_steps=10, cfg_strength=0.7, chunk_size=15,
                 use_tova=True, cache_max_size=512, head_weight_strategy="weighted", num_heads=8):
        super(AudioDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        self.sample_rate = sample_rate
        self.num_flow_steps = num_flow_steps
        self.cfg_strength = cfg_strength
        self.chunk_size = chunk_size
        
        # TOVA compression settings
        self.use_tova = use_tova
        self.cache_max_size = cache_max_size
        self.head_weight_strategy = head_weight_strategy
        self.num_heads = num_heads
        
        # Initialize TOVA compressor if enabled
        if use_tova:
            self.tova_compressor = TOVACompression(
                cache_max_size=cache_max_size,
                layer_based=True,
                head_weight_strategy=head_weight_strategy,
                num_heads=num_heads
            )
        
        # TOVA-related fields for KV caching
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        self.token_entropies = []
        
        # Upsampling to match Mel spectrogram frame rate (typically 50Hz)
        self.lookahead_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=5,  # Look ahead size of 4 + 1
            padding=4,      # Right padding for lookahead
            padding_mode='zeros'
        )
        
        # Causal upsampling transformer
        self.upsampling_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Upsampling projection
        self.upsample_proj = nn.Linear(hidden_dim, hidden_dim*2)
        
        # Conditional Flow Matching UNet
        self.unet = ChunkAwareFlowMatchingUNet(
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            num_heads=8
        )
        
        # Final projection to Mel spectrogram
        self.final_proj = nn.Linear(hidden_dim, mel_dim)
        
        # Vocoder for converting Mel spectrograms to waveforms
        self.vocoder = MelToWaveformVocoder(mel_dim=mel_dim, sample_rate=sample_rate)
        
    def create_attention_mask(self, mask_type, seq_len):
        """Create different attention masks for chunk-aware processing"""
        if mask_type == "non-causal":
            # Full attention - no masking
            return torch.ones(seq_len, seq_len, dtype=torch.bool)
        elif mask_type == "full-causal":
            # Strictly causal - only attend to past
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            return mask
        elif mask_type == "chunk-M":
            # Attend to past and M future frames
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            # Add diagonal bands for the future context
            for i in range(1, self.chunk_size + 1):
                mask = mask | torch.diag(torch.ones(seq_len - i, dtype=torch.bool), i)
            return mask
        elif mask_type == "chunk-2M":
            # Attend to past and 2*M future frames
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            # Add diagonal bands for the future context
            for i in range(1, 2 * self.chunk_size + 1):
                mask = mask | torch.diag(torch.ones(seq_len - i, dtype=torch.bool), i)
            return mask
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
    
    def forward(self, latent_patches, reference_audio=None, speaker_embedding=None, streaming=False):
        """
        Convert latent patches to audio waveform
        
        Args:
            latent_patches: Tensor of shape [batch_size, num_patches, input_dim]
            reference_audio: Optional reference audio for speaker characteristics
            speaker_embedding: Optional speaker embedding for voice characteristics
            streaming: Whether to use streaming mode with chunk-aware processing
            
        Returns:
            audio_waveform: Generated audio waveform
        """
        batch_size, num_patches, _ = latent_patches.shape
        
        # Apply lookahead convolution for future context
        x = latent_patches.transpose(1, 2)  # [B, D, T]
        x = self.lookahead_conv(x)
        x = x.transpose(1, 2)  # [B, T, D]
        
        # Upsample to match Mel spectrogram frame rate (2x for 25Hz -> 50Hz)
        if streaming:
            # Use causal mask for streaming mode
            mask_type = "chunk-M" if num_patches <= self.chunk_size else "chunk-2M"
            attn_mask = self.create_attention_mask(mask_type, num_patches)
            x = self.upsampling_transformer(x, mask=attn_mask)
        else:
            # Use non-causal mask for offline mode
            x = self.upsampling_transformer(x)
        
        # Upsample by factor of 2
        x = self.upsample_proj(x)
        x = x.reshape(batch_size, num_patches, 2, -1)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, num_patches * 2, -1)
        
        # Extract Mel features from reference audio if provided
        ref_mel = None
        if reference_audio is not None:
            ref_mel = self.extract_mel_features(reference_audio)
        
        # Apply flow matching to generate Mel spectrogram
        mel_spec = self.generate_mel_with_flow_matching(x, ref_mel, speaker_embedding, streaming)
        
        # Convert Mel spectrogram to waveform
        audio_waveform = self.vocoder(mel_spec)
        
        return audio_waveform
    
    def extract_mel_features(self, audio):
        """Extract Mel spectrogram features from audio"""
        # This would typically use a feature extractor like librosa or torchaudio
        # For simplicity, we'll assume audio is already in the right format
        return audio
    
    def generate_mel_with_flow_matching(self, upsampled_tokens, ref_mel=None, speaker_embedding=None, streaming=False):
        """Generate Mel spectrogram using conditional flow matching"""
        batch_size, seq_len, _ = upsampled_tokens.shape
        
        # Initialize from Gaussian noise
        x_t = torch.randn(batch_size, seq_len, self.mel_dim, device=upsampled_tokens.device)
        
        # Create masked reference Mel if provided
        x_masked = None
        if ref_mel is not None:
            # Randomly mask 70-100% of frames for training
            # For inference, use the reference Mel as is
            x_masked = ref_mel
        
        # Cosine scheduler for timesteps
        timesteps = torch.linspace(0, 1, self.num_flow_steps + 1)[1:]
        
        # Flow matching inference
        for t in timesteps:
            t_batch = torch.ones(batch_size, device=upsampled_tokens.device) * t
            
            # Predict noise
            noise_pred = self.unet(
                x_t, 
                t_batch, 
                upsampled_tokens, 
                x_masked, 
                speaker_embedding,
                streaming=streaming
            )
            
            # Apply classifier-free guidance if reference is provided
            if ref_mel is not None and speaker_embedding is not None:
                # Predict unconditional noise
                noise_pred_uncond = self.unet(
                    x_t, 
                    t_batch, 
                    upsampled_tokens,
                    None,
                    None,
                    streaming=streaming
                )
                
                # Apply guidance
                noise_pred = (1 + self.cfg_strength) * noise_pred - self.cfg_strength * noise_pred_uncond
            
            # Update x_t using the ODE solver step
            x_t = self.ode_step(x_t, noise_pred, t)
        
        # Final projection to Mel spectrogram
        mel_spec = self.final_proj(x_t)
        
        return mel_spec
    
    def ode_step(self, x_t, noise_pred, t):
        """Optimal Transport ODE step for flow matching"""
        # Simple Euler step for ODE solving
        dt = 1.0 / self.num_flow_steps
        return x_t + noise_pred * dt
        
    def enable_kv_caching(self):
        """Enable KV caching in attention modules for TOVA compression"""
        if not self.use_tova:
            return
            
        # Enable caching in UNet encoder and decoder blocks
        for block in self.unet.encoder_blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'forward'):
                # Store original forward method
                if not hasattr(block.attn, '_original_forward'):
                    block.attn._original_forward = block.attn.forward
                    
                    # Create reference to self for closure
                    audio_decoder_ref = self
                    
                    # Override forward method to track attention weights and cache KV
                    def new_forward(self_attn, query, key, value, key_padding_mask=None, 
                                   need_weights=True, attn_mask=None):
                        # Call original method with need_weights=True to get attention weights
                        attn_output, attn_weights = self_attn._original_forward(
                            query, key, value, 
                            key_padding_mask=key_padding_mask,
                            need_weights=True,
                            attn_mask=attn_mask
                        )
                        
                        # Store attention weights for TOVA
                        audio_decoder_ref.attention_weights = attn_weights
                        
                        # Update KV cache
                        if audio_decoder_ref.use_tova:
                            if audio_decoder_ref.k_cache is None:
                                audio_decoder_ref.k_cache = key
                                audio_decoder_ref.v_cache = value
                            else:
                                # Append to cache
                                audio_decoder_ref.k_cache = torch.cat([audio_decoder_ref.k_cache, key], dim=1)
                                audio_decoder_ref.v_cache = torch.cat([audio_decoder_ref.v_cache, value], dim=1)
                        
                        return attn_output, attn_weights
                    
                    # Bind new method to attention module
                    import types
                    block.attn.forward = types.MethodType(new_forward, block.attn)
        
        print("KV caching enabled for TOVA compression in AudioDecoder")
    
    def reset_cache(self):
        """Reset KV cache and entropy tracking"""
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        self.token_entropies = []
    
    def apply_tova_compression(self):
        """Apply TOVA compression to the KV cache if it exceeds the maximum size"""
        if not self.use_tova or self.k_cache is None or self.v_cache is None:
            return
            
        # Check if compression is needed
        if self.k_cache.size(1) <= self.cache_max_size:
            return
            
        # Apply basic TOVA compression if no entropy data
        if not self.token_entropies:
            self.k_cache, self.v_cache = self.tova_compressor(
                self.attention_weights,
                self.k_cache,
                self.v_cache
            )
        else:
            # Apply entropy-enhanced TOVA compression
            # Combine all token entropy values
            all_entropies = torch.cat(self.token_entropies)
            
            # Ensure entropy tensor matches KV cache size
            if all_entropies.size(0) < self.k_cache.size(1):
                # Pad with mean entropy
                mean_entropy = all_entropies.mean()
                padding = torch.full((self.k_cache.size(1) - all_entropies.size(0),), 
                                    mean_entropy, device=all_entropies.device)
                all_entropies = torch.cat([all_entropies, padding])
            
            # Apply compression with entropy data
            self.k_cache, self.v_cache = self.tova_compressor.compress_with_entropy(
                self.attention_weights,
                self.k_cache,
                self.v_cache,
                all_entropies[:self.k_cache.size(1)]
            )
            
        print(f"Applied TOVA compression. New KV cache size: {self.k_cache.size(1)}")
    
    def get_tova_stats(self):
        """Get statistics about TOVA compression"""
        if not self.use_tova:
            return {"tova_enabled": False}
            
        stats = self.tova_compressor.get_stats()
        
        # Add additional decoder-specific stats
        if self.k_cache is not None:
            stats["current_kv_cache_size"] = self.k_cache.size(1)
            stats["max_kv_cache_size"] = self.cache_max_size
            stats["kv_cache_memory_mb"] = (
                self.k_cache.element_size() * self.k_cache.nelement() +
                self.v_cache.element_size() * self.v_cache.nelement()
            ) / (1024 * 1024)
            
        return stats


class ChunkAwareFlowMatchingUNet(nn.Module):
    """
    UNet architecture for conditional flow matching with chunk-aware processing.
    Implements the causal convolutional Transformer UNet from CosyVoice 2.
    """
    def __init__(self, hidden_dim, mel_dim, num_heads=8):
        super(ChunkAwareFlowMatchingUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Speaker embedding projection
        self.speaker_proj = nn.Linear(512, hidden_dim)  # Assuming speaker embedding dim is 512
        
        # Input projection
        self.input_proj = nn.Linear(mel_dim, hidden_dim)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads) for _ in range(4)
        ])
        
        # Middle block
        self.middle_block = TransformerEncoderBlock(hidden_dim, num_heads)
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, num_heads) for _ in range(4)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, mel_dim)
    
    def forward(self, x, t, tokens, ref_mel=None, speaker_embedding=None, streaming=False):
        """
        Forward pass of the UNet
        
        Args:
            x: Input tensor [B, T, mel_dim]
            t: Timestep tensor [B]
            tokens: Upsampled token embeddings [B, T, hidden_dim]
            ref_mel: Optional reference Mel spectrogram
            speaker_embedding: Optional speaker embedding
            streaming: Whether to use streaming mode
            
        Returns:
            Predicted noise
        """
        # Project input to hidden dimension
        h = self.input_proj(x)
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Add time embedding
        h = h + t_emb.unsqueeze(1)
        
        # Add token embeddings
        h = h + tokens
        
        # Add speaker embedding if provided
        if speaker_embedding is not None:
            spk_emb = self.speaker_proj(speaker_embedding)
            h = h + spk_emb.unsqueeze(1)
        
        # Add reference Mel if provided
        if ref_mel is not None:
            ref_proj = self.input_proj(ref_mel)
            h = h + ref_proj
        
        # Encoder
        skips = []
        for block in self.encoder_blocks:
            h = block(h, streaming=streaming)
            skips.append(h)
        
        # Middle
        h = self.middle_block(h, streaming=streaming)
        
        # Decoder with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = block(h, skip, streaming=streaming)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with optional chunk-aware processing"""
    def __init__(self, hidden_dim, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, streaming=False):
        # Self-attention with optional causal mask for streaming
        if streaming:
            # Create causal mask for streaming mode
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
            
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        else:
            h = self.norm1(x)
            h, _ = self.attn(h, h, h)
        
        x = x + h
        
        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        
        return x + h


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with skip connections and optional chunk-aware processing"""
    def __init__(self, hidden_dim, num_heads):
        super(TransformerDecoderBlock, self).__init__()
        self.attn1 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, skip, streaming=False):
        # Self-attention with optional causal mask for streaming
        if streaming:
            # Create causal mask for streaming mode
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
            
            h = self.norm1(x)
            h, _ = self.attn1(h, h, h, attn_mask=causal_mask)
        else:
            h = self.norm1(x)
            h, _ = self.attn1(h, h, h)
        
        x = x + h
        
        # Cross-attention with skip connection
        h = self.norm2(x)
        h, _ = self.attn2(h, skip, skip)
        
        x = x + h
        
        # FFN
        h = self.norm3(x)
        h = self.ffn(h)
        
        return x + h


class MelToWaveformVocoder(nn.Module):
    """
    Vocoder for converting Mel spectrograms to audio waveforms.
    Based on WaveNext from CosyVoice 2 paper.
    """
    def __init__(self, mel_dim=80, sample_rate=24000):
        super(MelToWaveformVocoder, self).__init__()
        self.mel_dim = mel_dim
        self.sample_rate = sample_rate
        
        # Simplified vocoder architecture
        # In a real implementation, this would be a more sophisticated model like WaveNext
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(mel_dim, 512, kernel_size=16, stride=8, padding=4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 1, kernel_size=8, stride=2, padding=3),
            nn.Tanh()
        )
    
    def forward(self, mel_spec):
        """Convert Mel spectrogram to audio waveform"""
        # Transpose for 1D convolution
        x = mel_spec.transpose(1, 2)
        
        # Upsample to audio sample rate
        waveform = self.upsample(x)
        
        # Reshape to [batch_size, audio_length]
        waveform = waveform.squeeze(1)
        
        return waveform

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
