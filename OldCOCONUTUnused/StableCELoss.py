#For Softmax Or Cross-Entropy Loss training. Improves Grokking Chances. 

import torch
import torch.nn.functional as F

def stable_s(x):
    """Numerically stable version of s(x)."""
    return torch.where(x >= 0, x + 1, 1 / (1 - x))

def stable_softmax(x, dim=-1):
    """Numerically stable StableMax."""
    s_x = stable_s(x)
    return s_x / s_x.sum(dim=dim, keepdim=True)

def stable_cross_entropy_loss(logits, labels):
    """
    Stable Cross-Entropy Loss using StableMax.

    Args:
        logits (torch.Tensor): Model output logits.
        labels (torch.Tensor): True labels (LongTensor).

    Returns:
        torch.Tensor: Loss value.
    """
    log_probs = torch.log(stable_softmax(logits, dim=-1))  # Apply StableMax and then log
    nll_loss = F.nll_loss(log_probs, labels) # Negative log likelihood loss
    return nll_loss
