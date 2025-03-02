import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from TOVACompression import TOVACompression

def test_standard_compression():
    """Test the standard TOVA compression without entropy enhancement."""
    print("\n=== Testing Standard TOVA Compression ===")
    
    # Create a simulated attention pattern with 4 heads and 1000 tokens
    num_heads = 4
    num_tokens = 1000
    cache_size = 256
    
    # Create synthetic attention weights - different pattern for each head
    attn_weights = torch.zeros(num_heads, 1, num_tokens)
    
    # Head 1: Focuses on early tokens (prompt/context tokens)
    early_indices = torch.arange(100)
    attn_weights[0, 0, early_indices] = torch.softmax(torch.randn(100) * 2, dim=0)
    
    # Head 2: Focuses on recent tokens (recency bias)
    recent_indices = torch.arange(num_tokens - 200, num_tokens)
    attn_weights[1, 0, recent_indices] = torch.softmax(torch.randn(200) * 2, dim=0)
    
    # Head 3: Focuses on important semantic tokens (simulated by random spikes)
    semantic_indices = torch.randint(0, num_tokens, (50,))
    attn_weights[2, 0, semantic_indices] = torch.softmax(torch.randn(50) * 3, dim=0)
    
    # Head 4: General attention across the sequence
    attn_weights[3, 0, :] = torch.softmax(torch.randn(num_tokens) * 0.5, dim=0)
    
    # Always ensure first token has some attention
    attn_weights[:, 0, 0] = 0.1
    
    # Normalize each head's attention
    for h in range(num_heads):
        attn_weights[h, 0, :] = attn_weights[h, 0, :] / attn_weights[h, 0, :].sum()
    
    # Create key and value cache
    hidden_dim = 64
    k_cache = torch.randn(num_heads, num_tokens, hidden_dim)
    v_cache = torch.randn(num_heads, num_tokens, hidden_dim)
    
    # Create TOVA compressor with different head weighting strategies
    strategies = ["mean", "max", "weighted"]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} head weighting strategy:")
        compressor = TOVACompression(
            cache_max_size=cache_size,
            layer_based=True,
            head_weight_strategy=strategy,
            num_heads=num_heads,
            learning_rate=0.05,
            weight_momentum=0.9
        )
        
        # Run compression multiple times to simulate multiple forwards
        num_iterations = 50
        
        for i in range(num_iterations):
            # Add small random noise to attention for variety
            noise = torch.randn_like(attn_weights) * 0.01
            noisy_attn = attn_weights + noise
            for h in range(num_heads):
                noisy_attn[h, 0, :] = torch.softmax(noisy_attn[h, 0, :], dim=0)
            
            # Apply compression
            compressed_k, compressed_v = compressor(noisy_attn, k_cache, v_cache)
            
            # Print stats occasionally
            if (i + 1) % 10 == 0:
                stats = compressor.get_stats()
                print(f"  Iteration {i+1}: Compression ratio: {stats['average_compression_ratio']:.4f}")
                
                if strategy == "weighted" and hasattr(compressor, 'head_weights'):
                    weights = torch.softmax(compressor.head_weights, dim=0).detach().cpu().numpy()
                    print(f"  Head weights: {', '.join([f'{w:.3f}' for w in weights])}")
        
        # Store results
        results[strategy] = {
            "compressor": compressor,
            "final_k": compressed_k,
            "final_v": compressed_v,
            "stats": compressor.get_stats()
        }
        
        # For weighted strategy, plot weight evolution
        if strategy == "weighted" and hasattr(compressor, 'plot_weight_evolution'):
            os.makedirs("tova_test_results", exist_ok=True)
            compressor.plot_weight_evolution(save_path="tova_test_results/head_weights_evolution.png")
            print(f"  Head weight evolution plot saved to tova_test_results/head_weights_evolution.png")
    
    # Compare compression results
    print("\nCompression Results Comparison:")
    for strategy, result in results.items():
        stats = result["stats"]
        print(f"{strategy.capitalize()} Strategy:")
        print(f"  - Compression ratio: {stats['average_compression_ratio']:.4f}")
        print(f"  - Average compression time: {stats['compression_time_ms']:.2f} ms")
        print(f"  - Original tokens: {num_tokens}, Compressed tokens: {cache_size}")
        
    return results


def test_entropy_enhanced_compression():
    """Test TOVA compression with entropy enhancement."""
    print("\n=== Testing Entropy-Enhanced TOVA Compression ===")
    
    # Create a simulated attention pattern with 4 heads and 1000 tokens
    num_heads = 4
    num_tokens = 1000
    cache_size = 256
    
    # Create synthetic attention weights (similar to the standard test)
    attn_weights = torch.zeros(num_heads, 1, num_tokens)
    
    # Set up attention patterns for different heads
    # Head 1: Focuses on early tokens
    attn_weights[0, 0, :100] = torch.softmax(torch.randn(100) * 2, dim=0)
    
    # Head 2: Focuses on recent tokens
    attn_weights[1, 0, -200:] = torch.softmax(torch.randn(200) * 2, dim=0)
    
    # Head 3: Random semantic tokens
    semantic_indices = torch.randint(0, num_tokens, (50,))
    attn_weights[2, 0, semantic_indices] = torch.softmax(torch.randn(50) * 3, dim=0)
    
    # Head 4: General attention
    attn_weights[3, 0, :] = torch.softmax(torch.randn(num_tokens) * 0.5, dim=0)
    
    # Normalize each head's attention
    for h in range(num_heads):
        attn_weights[h, 0, :] = attn_weights[h, 0, :] / attn_weights[h, 0, :].sum()
    
    # Create key and value cache
    hidden_dim = 64
    k_cache = torch.randn(num_heads, num_tokens, hidden_dim)
    v_cache = torch.randn(num_heads, num_tokens, hidden_dim)
    
    # Create synthetic entropy values
    # We'll create high entropy for some tokens that have low attention
    # to demonstrate how entropy can change token selection
    entropy_values = torch.zeros(num_tokens)
    
    # Base entropy on token position (middle tokens have higher entropy in this example)
    position_entropy = torch.sin(torch.linspace(0, 3*np.pi, num_tokens)) * 0.5 + 0.5
    
    # Add some random variation
    random_entropy = torch.rand(num_tokens) * 0.5
    entropy_values = position_entropy + random_entropy
    
    # Make a few high-entropy semantic tokens
    high_entropy_tokens = torch.randint(0, num_tokens, (30,))
    entropy_values[high_entropy_tokens] = 0.9 + torch.rand(30) * 0.1
    
    # Create TOVA compressor with entropy enhancement
    compressor = TOVACompression(
        cache_max_size=cache_size,
        layer_based=True,
        head_weight_strategy="weighted",
        num_heads=num_heads,
        learning_rate=0.05,
        weight_momentum=0.9,
        entropy_weight=0.4  # Weight for entropy values
    )
    
    # Run standard compression first (without entropy)
    print("\nRunning standard compression (without entropy):")
    compressed_k_std, compressed_v_std = compressor(attn_weights, k_cache, v_cache)
    std_indices = torch.zeros(num_tokens, dtype=torch.bool)
    for idx in range(compressed_k_std.size(1)):
        # Find the token that matches the compressed token
        for orig_idx in range(num_tokens):
            if torch.all(k_cache[:, orig_idx, :] == compressed_k_std[:, idx, :]):
                std_indices[orig_idx] = True
                break
    
    # Reset compressor stats
    compressor.reset_stats()
    
    # Run entropy-enhanced compression
    print("\nRunning entropy-enhanced compression:")
    compressed_k_ent, compressed_v_ent = compressor.compress_with_entropy(
        attn_weights, k_cache, v_cache, entropy_values
    )
    ent_indices = torch.zeros(num_tokens, dtype=torch.bool)
    for idx in range(compressed_k_ent.size(1)):
        # Find the token that matches the compressed token
        for orig_idx in range(num_tokens):
            if torch.all(k_cache[:, orig_idx, :] == compressed_k_ent[:, idx, :]):
                ent_indices[orig_idx] = True
                break
    
    # Calculate tokens that were kept by entropy but not by standard compression
    entropy_only = torch.logical_and(ent_indices, ~std_indices)
    print(f"Tokens kept by entropy-enhanced but not by standard: {entropy_only.sum().item()}")
    
    # Plot entropy values vs attention scores to visualize the difference
    plt.figure(figsize=(12, 6))
    
    # Plot attention scores (average across heads)
    avg_attn = attn_weights.mean(dim=0).squeeze().numpy()
    plt.subplot(1, 2, 1)
    plt.plot(avg_attn, label='Avg Attention', alpha=0.7)
    plt.plot(entropy_values.numpy(), label='Entropy', alpha=0.7)
    plt.legend()
    plt.title('Attention vs Entropy Values')
    plt.xlabel('Token Position')
    plt.ylabel('Score')
    
    # Plot token selection differences
    plt.subplot(1, 2, 2)
    plt.scatter(range(num_tokens), avg_attn, c='blue', alpha=0.3, label='All Tokens')
    plt.scatter(
        np.where(std_indices.numpy())[0], 
        avg_attn[std_indices.numpy()], 
        c='green', alpha=0.7, label='Standard Kept'
    )
    plt.scatter(
        np.where(entropy_only.numpy())[0], 
        avg_attn[entropy_only.numpy()], 
        c='red', alpha=0.7, label='Entropy Only'
    )
    plt.legend()
    plt.title('Token Selection Differences')
    plt.xlabel('Token Position')
    plt.ylabel('Attention Score')
    
    plt.tight_layout()
    os.makedirs("tova_test_results", exist_ok=True)
    plt.savefig("tova_test_results/entropy_comparison.png")
    plt.close()
    print("Entropy comparison plot saved to: tova_test_results/entropy_comparison.png")
    
    # Plot head weight evolution
    compressor.plot_weight_evolution(save_path="tova_test_results/entropy_head_weights.png")
    print("Head weight evolution plot saved to: tova_test_results/entropy_head_weights.png")
    
    # Return both compressions for comparison
    return {
        "standard": {
            "k_cache": compressed_k_std,
            "v_cache": compressed_v_std,
            "kept_indices": std_indices
        },
        "entropy_enhanced": {
            "k_cache": compressed_k_ent,
            "v_cache": compressed_v_ent,
            "kept_indices": ent_indices,
            "entropy_values": entropy_values
        }
    }

if __name__ == "__main__":
    print("Starting TOVA compression tests...")
    
    # Test standard compression
    standard_results = test_standard_compression()
    
    # Test entropy enhanced compression
    entropy_results = test_entropy_enhanced_compression()
    
    print("\nAll tests completed successfully!")
    print("Check the tova_test_results directory for visualization plots.")
