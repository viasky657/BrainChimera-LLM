"""
RedPajama Dataset Example for CROW Backdoor Elimination

This example demonstrates how to use the RedPajama dataset with the CROW backdoor elimination
method to purify a language model of potential backdoor attacks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Crow import apply_crow_training, get_redpajama_dataset

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer 
    # For demonstration, we use a small model
    # In a real application, you would use your specific model that needs backdoor elimination
    model_name = "gpt2-medium"  # This is just an example - use your actual model
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    # Option 1: Load the RedPajama dataset directly
    print("Option 1: Load RedPajama dataset directly")
    redpajama_dataset = get_redpajama_dataset(max_chars=1000)
    
    # Prepare dataset for the model by tokenizing the text
    tokenized_dataset = redpajama_dataset.prepare_for_model(tokenizer)
    print(f"Prepared RedPajama dataset with {len(tokenized_dataset)} samples")
    
    # Option 2: Let apply_crow_training handle the RedPajama dataset loading
    print("\nOption 2: Let apply_crow_training handle the RedPajama dataset")
    purified_model, metrics = apply_crow_training(
        model=model,
        train_data=None,  # No dataset provided, will use RedPajama
        epsilon=0.1,
        alpha=5.5,
        learning_rate=2e-5,
        num_epochs=1,  # Using just 1 epoch for demonstration
        batch_size=2,
        device=device,
        use_redpajama=True,
        tokenizer=tokenizer
    )
    
    print("\nCROW training completed successfully!")
    
    # The model has now been purified of potential backdoors
    # You can now use the purified model for your tasks
    
    # Example: Generate text with the purified model
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nGenerating text with prompt: '{prompt}'")
    with torch.no_grad():
        outputs = purified_model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
