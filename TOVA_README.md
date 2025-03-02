# TOVA Compression with Weighted Head Strategy

## Overview

This implementation enhances the COCONUT model with TOVA (Token Omission Via Attention) compression from the "Transformers are Multi-State RNNs" paper, with our own modifications to support dynamic binary patches and entropy-based token importance.

## Key Features

### 1. Weighted Head Strategy
- Implemented learnable weights for attention heads to prioritize the most informative heads
- Automatic weight updates based on head contribution to token selection
- Support for different head weighting strategies: mean, max, and weighted

### 2. Entropy-Enhanced Compression
- Token importance based on both attention scores and information entropy
- Dynamically combines attention patterns with entropy values for more intelligent token retention
- Preserves high-information-content tokens even with lower attention scores

### 3. Visualization and Monitoring
- Tools to track head weight evolution over time
- Debug mode to monitor compression statistics
- Plotting functionality to visualize learning progress

## Implementation Details

The implementation spans two main files:

### TOVACompression.py
- Core compression algorithm with weighted head scoring
- Learning mechanism that updates head weights based on correlation with token selection
- Support for both standard and entropy-enhanced compression modes
- Visualization tools for weight evolution tracking

### COCONUTWLatentThinking.py
- Integration with MultiModalEncoder for multimodal compression
- Debug mode support for detailed logging
- Head weight visualization method
- Entropy tracking in encoder components

## Usage Example

```python
# Initialize MultiModalEncoder with weighted head strategy
encoder = MultiModalEncoder(
    vocab_size=256, 
    embed_dim=768, 
    sonar_dim=512, 
    patch_dim=768,
    audio_entropy_predictor=entropy_predictor,
    cache_max_size=512,
    use_tova=True,
    head_weight_strategy="weighted",
    num_heads=4,
    debug_mode=True
)

# Process inputs through encoders
encoded_output = encoder(input_data)

# Apply compression to all encoders
encoder.compress_all_encoders()

# Visualize head weight evolution
encoder.visualize_head_weights(save_path="head_weights.png")
```

## Benefits

1. **Memory Efficiency**: Reduces KV cache size by up to 87.5% (1/8th of original)
2. **Performance Retention**: Maintains model performance with dramatically smaller cache
3. **Computation Speed**: Increases throughput by reducing memory bottlenecks
4. **Adaptive Learning**: Automatically learns optimal compression strategy during use
5. **Multimodal Support**: Works across different modalities (text, audio, video, PDF)

## Technical Implementation Notes

The weighted head strategy is implemented by tracking each head's contribution to token selection decisions and updating weights to prioritize heads whose attention patterns correlate strongly with effective token selection. The entropy enhancement further improves this by factoring in the information content of tokens, ensuring high-information tokens are retained even if they have lower attention scores.

The compression is fully dynamic and adapts to the content being processed, with no need for pre-training or fine-tuning specific to the compression mechanism.
