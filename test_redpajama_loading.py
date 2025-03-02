"""
Test Script for RedPajama Dataset Loading

This script simply tests the loading of the RedPajama dataset to verify that the
integration is working correctly, without running the full CROW training process.
"""

import torch
from transformers import AutoTokenizer
from Crow import get_redpajama_dataset

def main():
    print("Testing RedPajama dataset loading for CROW...")
    
    # Load a small amount of RedPajama data (only 500 characters for testing)
    print("\nLoading RedPajama dataset (500 characters)...")
    redpajama_dataset = get_redpajama_dataset(max_chars=500)
    
    # Display basic dataset information
    print(f"\nLoaded dataset with {len(redpajama_dataset)} samples")
    
    # Display the text content of the samples
    print("\nSample texts from the RedPajama dataset:")
    for i, sample in enumerate(redpajama_dataset.data):
        print(f"\nSample {i+1}:")
        text = sample["text"]
        # Print only the first 200 characters if the text is longer
        if len(text) > 200:
            print(f"{text[:200]}... (truncated, total length: {len(text)} chars)")
        else:
            print(text)
        
        # Only show the first 3 samples to avoid excessive output
        if i >= 2:
            remaining = len(redpajama_dataset) - 3
            if remaining > 0:
                print(f"\n... and {remaining} more sample(s)")
            break
    
    # Now test tokenization with a common tokenizer
    print("\nTesting dataset tokenization...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Tokenizing texts...")
        tokenized_dataset = redpajama_dataset.prepare_for_model(tokenizer)
        
        print(f"\nTokenized dataset has {len(tokenized_dataset)} samples")
        
        # Print token counts for each sample
        print("\nToken counts per sample:")
        for i, sample in enumerate(tokenized_dataset.data):
            print(f"Sample {i+1}: {len(sample['input_ids'])} tokens")
            
            # Only show the first 3 samples
            if i >= 2:
                remaining = len(tokenized_dataset) - 3
                if remaining > 0:
                    print(f"... and {remaining} more sample(s)")
                break
        
        print("\nTokenization test successful!")
    except Exception as e:
        print(f"Error during tokenization: {e}")
    
    print("\nRedPajama dataset loading test completed!")

if __name__ == "__main__":
    main()
