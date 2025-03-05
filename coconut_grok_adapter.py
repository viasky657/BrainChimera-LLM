import torch
from GrokOptimizers import use_grokking_optimizations, StableMax, StableCrossEntropyLoss, OrthoAdamW

def configure_coconut_for_grokking(model, lr=1e-3, weight_decay=0.01, use_stablemax=True):
    """
    Configure a COCONUT model to use the optimizations from the paper
    "Grokking at the Edge of Numerical Stability" to help it generalize earlier.
    
    Args:
        model: The COCONUT model to optimize
        lr: Learning rate for the optimizer
        weight_decay: Weight decay coefficient
        use_stablemax: Whether to replace CrossEntropyLoss with StableCrossEntropyLoss
        
    Returns:
        The configured model with updated optimizer and loss function
    """
    print("Configuring COCONUT model for improved grokking...")
    
    optim_kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
        'betas': (0.9, 0.999)
    }
    
    loss_kwargs = {
        'reduction': 'mean'  # Use mean reduction by default
    }
    
    # Apply grokking optimizations
    result = use_grokking_optimizations(
        model,
        loss=use_stablemax,
        optimizer=True,
        optimizer_type="OrthoAdamW",  # Best optimizer for grokking according to the paper
        optim_kwargs=optim_kwargs,
        loss_kwargs=loss_kwargs
    )
    
    if use_stablemax:
        loss_fn, optimizer = result
        print("Applied StableCrossEntropyLoss and OrthoAdamW optimizer")
    else:
        optimizer = result
        print("Applied OrthoAdamW optimizer (keeping original loss function)")

    # Monkey patch the forward method to use StableMax for any softmax operations
    # This helps prevent Softmax Collapse during inference as well
    if hasattr(model, 'forward') and use_stablemax:
        original_forward = model.forward
        stablemax = StableMax()
        
        def forward_with_stablemax(*args, **kwargs):
            # Call the original forward method
            result = original_forward(*args, **kwargs)
            
            # If the result includes logits, apply StableMax instead of Softmax
            if isinstance(result, tuple) and len(result) >= 1:
                # Apply StableMax to logits if they're part of the return value
                if hasattr(result, 'logits') and result.logits is not None:
                    # Replace softmax with stablemax for the logits
                    result = result._replace(logits=stablemax(result.logits))
            
            return result
        
        # Replace the forward method
        model.forward = forward_with_stablemax
        print("Patched forward method to use StableMax for softmax operations")
    
    # Return the model for chaining
    return model


def train_coconut_with_grokking(model, dataloader, num_epochs=10, lr=1e-3, weight_decay=0.01):
    """
    Train a COCONUT model using the optimizations from the paper to achieve faster grokking.
    
    Args:
        model: The COCONUT model to train
        dataloader: DataLoader containing the training data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Training metrics including loss and accuracy history
    """
    # Configure the model for grokking
    model = configure_coconut_for_grokking(model, lr, weight_decay)
    
    # Get the optimizer (it might have been set on the model)
    optimizer = model.optimizer if hasattr(model, 'optimizer') else None
    
    # If optimizer is still None, create a new one
    if optimizer is None:
        optimizer = OrthoAdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get the loss function (it might have been set on the model)
    criterion = (model.criterion if hasattr(model, 'criterion') else 
                (model.loss_fn if hasattr(model, 'loss_fn') else None))
    
    # If criterion is still None, create a new StableCrossEntropyLoss
    if criterion is None:
        criterion = StableCrossEntropyLoss()
    
    # Track metrics
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i, batch in enumerate(dataloader):
            # Process batch based on expected format
            # This is a generic example - adjust for your specific data format
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
            else:
                # Assume batch is a dictionary with 'input_ids' and 'labels'
                inputs = batch.get('input_ids', batch.get('inputs', batch))
                targets = batch.get('labels', batch.get('targets', None))
            
            # Forward pass
            outputs = model(inputs)
            
            # Extract logits from outputs based on model output format
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                # Assuming first element contains logits or has a logits attribute
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
            else:
                # Assume outputs are already logits
                logits = outputs
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            
            # Calculate accuracy if possible
            if hasattr(logits, 'argmax') and hasattr(targets, 'size'):
                pred = logits.argmax(dim=1, keepdim=True)
                if targets.dim() > 1 and targets.size(1) > 1:
                    # One-hot encoded targets
                    target_indices = targets.argmax(dim=1, keepdim=True)
                else:
                    # Class indices
                    target_indices = targets.view_as(pred)
                correct += pred.eq(target_indices).sum().item()
                total += targets.size(0)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics['train_loss'].append(avg_loss)
        metrics['train_acc'].append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("Training complete!")
    return metrics