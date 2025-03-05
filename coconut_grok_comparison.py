import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from torch.utils.data import DataLoader, TensorDataset

from GrokOptimizers import StableCrossEntropyLoss, OrthoAdamW, StableMax

# Create directory for saving results
os.makedirs("grok_results", exist_ok=True)

class SimpleNet(nn.Module):
    """
    A simple network for modular arithmetic tasks,
    similar to the models used in the grokking paper
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


def create_modular_arithmetic_dataset(operation="addition", modulus=113, train_percent=40):
    """
    Create a dataset for modular arithmetic, similar to the one used in the paper.
    
    Args:
        operation: "addition" or "multiplication"
        modulus: The modulus to use
        train_percent: Percentage of examples to use for training
        
    Returns:
        train_loader, test_loader: DataLoaders for training and test data
    """
    # Create all possible pairs
    all_pairs = []
    all_targets = []
    
    for a in range(modulus):
        for b in range(modulus):
            all_pairs.append([a, b])
            if operation == "addition":
                target = (a + b) % modulus
            elif operation == "multiplication":
                target = (a * b) % modulus
            all_targets.append(target)
    
    # Convert to one-hot encoding
    pairs_one_hot = []
    for a, b in all_pairs:
        # Create one-hot vectors
        a_one_hot = np.zeros(modulus)
        b_one_hot = np.zeros(modulus)
        a_one_hot[a] = 1
        b_one_hot[b] = 1
        # Concatenate
        pairs_one_hot.append(np.concatenate([a_one_hot, b_one_hot]))
    
    # Convert to PyTorch tensors
    all_pairs = torch.tensor(pairs_one_hot, dtype=torch.float32)
    all_targets = torch.tensor(all_targets, dtype=torch.long)
    
    # Split into train and test
    total_examples = all_pairs.shape[0]
    train_size = int(total_examples * train_percent / 100)
    
    # Use a fixed random seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(total_examples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_pairs = all_pairs[train_indices]
    train_targets = all_targets[train_indices]
    
    test_pairs = all_pairs[test_indices]
    test_targets = all_targets[test_indices]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_pairs, train_targets)
    test_dataset = TensorDataset(test_pairs, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"Created {operation} modulo {modulus} dataset")
    print(f"Train examples: {len(train_dataset)}, Test examples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, 
                      device, epochs=5000, early_stop_accuracy=0.95, checkpoint_freq=100):
    """
    Train the model and evaluate periodically, tracking metrics for analysis
    
    Args:
        model: The model to train
        train_loader, test_loader: DataLoaders for training and test data
        optimizer: The optimizer to use
        criterion: The loss function to use
        device: The device to train on
        epochs: Maximum number of epochs to train for
        early_stop_accuracy: Stop training if test accuracy exceeds this
        checkpoint_freq: How often to evaluate and record metrics
        
    Returns:
        Dictionary of training metrics
    """
    model.to(device)
    
    # Metrics to track
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch': [],
        'weight_norm': [],
        'time_elapsed': [],
        'checkpoint_epochs': []
    }
    
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Calculate weight norm
        weight_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        # Checkpoint evaluation
        if epoch % checkpoint_freq == 0 or epoch == epochs - 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            metrics['epoch'].append(epoch)
            metrics['weight_norm'].append(weight_norm)
            metrics['time_elapsed'].append(time.time() - start_time)
            metrics['checkpoint_epochs'].append(epoch)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"Weight Norm: {weight_norm:.4f}")
            
            # Early stopping
            if test_acc >= early_stop_accuracy:
                print(f"Early stopping at epoch {epoch} with test accuracy {test_acc:.4f}")
                break
    
    return metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on the dataloader"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(dataloader)
    test_acc = correct / total
    
    return test_loss, test_acc


def run_comparison_experiments():
    """
    Run several experiments comparing different optimizers 
    to demonstrate the benefits of the grokking optimizers
    """
    # Parameters
    input_dim = 226  # 113 * 2 for one-hot encoded pairs
    hidden_dim = 200
    output_dim = 113  # Modulus
    lr = 1e-3
    weight_decay = 0.01
    epochs = 5000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the dataset
    train_loader, test_loader = create_modular_arithmetic_dataset(
        operation="addition", 
        modulus=113, 
        train_percent=40
    )
    
    # Configure experiments
    experiments = [
        {
            "name": "Adam",
            "optimizer_class": optim.Adam,
            "optimizer_kwargs": {"lr": lr},
            "criterion_class": nn.CrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "blue"
        },
        {
            "name": "AdamW",
            "optimizer_class": optim.AdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
            "criterion_class": nn.CrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "green"
        },
        {
            "name": "OrthoAdamW",
            "optimizer_class": OrthoAdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
            "criterion_class": nn.CrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "red"
        },
        {
            "name": "OrthoAdamW + StableCrossEntropyLoss",
            "optimizer_class": OrthoAdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
            "criterion_class": StableCrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "purple"
        },
    ]
    
    # Run experiments
    all_metrics = {}
    
    for exp in experiments:
        print(f"\n=== Running experiment: {exp['name']} ===\n")
        
        # Create model
        model = SimpleNet(input_dim, hidden_dim, output_dim)
        
        # Create optimizer and criterion
        optimizer = exp["optimizer_class"](model.parameters(), **exp["optimizer_kwargs"])
        criterion = exp["criterion_class"](**exp["criterion_kwargs"])
        
        # Train and evaluate
        metrics = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            checkpoint_freq=100
        )
        
        all_metrics[exp["name"]] = {"metrics": metrics, "color": exp["color"]}
    
    # Plot results
    plot_comparison_results(all_metrics)


def plot_comparison_results(all_metrics):
    """Plot the comparison results"""
    plt.figure(figsize=(18, 13))
    
    # Plot 1: Test Accuracy vs Epoch
    plt.subplot(2, 2, 1)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        plt.plot(metrics["checkpoint_epochs"], metrics["test_acc"], 
                 label=name, marker='o', markersize=4, color=color)
    
    plt.title("Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Train vs Test Accuracy
    plt.subplot(2, 2, 2)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        epoch_reached_95 = None
        
        # Find epoch where test accuracy reached 0.95
        for i, acc in enumerate(metrics["test_acc"]):
            if acc >= 0.95:
                epoch_reached_95 = metrics["checkpoint_epochs"][i]
                break
        
        # Plot scatter point if epoch was found
        if epoch_reached_95 is not None:
            plt.scatter([epoch_reached_95], [0.95], 
                       s=100, color=color, marker='*',
                       label=f"{name} reached 95% at epoch {epoch_reached_95}")
    
    plt.title("Epochs to Reach 95% Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 3: Weight Norm vs Epoch
    plt.subplot(2, 2, 3)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        plt.plot(metrics["checkpoint_epochs"], metrics["weight_norm"], 
                 label=name, marker='o', markersize=4, color=color)
    
    plt.title("Weight Norm vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Norm")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 4: Train vs Test Accuracy Gap
    plt.subplot(2, 2, 4)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        gap = [train - test for train, test in zip(metrics["train_acc"], metrics["test_acc"])]
        plt.plot(metrics["checkpoint_epochs"], gap, 
                 label=name, marker='o', markersize=4, color=color)
    
    plt.title("Train-Test Accuracy Gap vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("grok_results/optimizer_comparison.png")
    plt.show()


def run_weight_decay_ablation():
    """
    Test the StableCrossEntropyLoss with OrthoAdamW without weight decay
    to demonstrate that it can achieve generalization without regularization
    """
    # Parameters
    input_dim = 226  # 113 * 2 for one-hot encoded pairs
    hidden_dim = 200
    output_dim = 113  # Modulus
    lr = 1e-3
    epochs = 5000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the dataset
    train_loader, test_loader = create_modular_arithmetic_dataset(
        operation="addition", 
        modulus=113, 
        train_percent=40
    )
    
    # Configure experiments testing different weight decay values with StableMax
    experiments = [
        {
            "name": "AdamW + WD=0.01",
            "optimizer_class": optim.AdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": 0.01},
            "criterion_class": nn.CrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "blue"
        },
        {
            "name": "AdamW + WD=0",
            "optimizer_class": optim.AdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": 0},
            "criterion_class": nn.CrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "green"
        },
        {
            "name": "OrthoAdamW + StableCE + WD=0",
            "optimizer_class": OrthoAdamW,
            "optimizer_kwargs": {"lr": lr, "weight_decay": 0},
            "criterion_class": StableCrossEntropyLoss,
            "criterion_kwargs": {},
            "color": "red"
        },
    ]
    
    # Run experiments
    all_metrics = {}
    
    for exp in experiments:
        print(f"\n=== Running experiment: {exp['name']} ===\n")
        
        # Create model
        model = SimpleNet(input_dim, hidden_dim, output_dim)
        
        # Create optimizer and criterion
        optimizer = exp["optimizer_class"](model.parameters(), **exp["optimizer_kwargs"])
        criterion = exp["criterion_class"](**exp["criterion_kwargs"])
        
        # Train and evaluate
        metrics = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            checkpoint_freq=100
        )
        
        all_metrics[exp["name"]] = {"metrics": metrics, "color": exp["color"]}
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Test Accuracy vs Epoch
    plt.subplot(1, 2, 1)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        plt.plot(metrics["checkpoint_epochs"], metrics["test_acc"], 
                 label=name, marker='o', markersize=4, color=color)
    
    plt.title("Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Weight Norm vs Epoch
    plt.subplot(1, 2, 2)
    for name, data in all_metrics.items():
        metrics = data["metrics"]
        color = data["color"]
        plt.plot(metrics["checkpoint_epochs"], metrics["weight_norm"], 
                 label=name, marker='o', markersize=4, color=color)
    
    plt.title("Weight Norm vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Norm")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("grok_results/weightdecay_ablation.png")
    plt.show()


if __name__ == "__main__":
    print("=== Grokking Optimizer Comparison ===")
    print("This script compares different optimizers for training on a modular arithmetic task.")
    print("It demonstrates how the StableMax and OrthoAdamW optimizers can help models")
    print("generalize (grok) earlier in training.\n")
    
    # Run the main comparison
    print("\n=== Running Main Optimizer Comparison ===\n")
    run_comparison_experiments()
    
    # Run the weight decay ablation study
    print("\n=== Running Weight Decay Ablation Study ===\n")
    run_weight_decay_ablation()
    
    print("\nExperiments complete. Results saved to grok_results/optimizer_comparison.png")
    print("and grok_results/weightdecay_ablation.png")