#Ortho Gradient Learning Optimizer Improves the rate at which the Model Groks (will take about half as much training to Grok or generalize its learning using this optimizer over a traditional one.)
import torch
from torch.optim import Optimizer

class OrthoGrad(Optimizer):
    """
    Orthogonal Gradient (⊥Grad) Optimizer.

    Wraps an existing optimizer and projects the gradient to be orthogonal
    to the current weight vector.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        defaults = {} # Inherit defaults from wrapped optimizer if needed
        super(OrthoGrad, self).__init__(optimizer.param_groups, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        group = self.param_groups[0] # Assuming single param group for simplicity

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            weight = p.data

            # Calculate orthogonal component of gradient (Equation 12 from paper)
            grad_ortho = grad - (torch.sum(weight * grad) / torch.sum(weight * weight)) * weight

            # Replace original gradient with orthogonal gradient for update
            p.grad.data = grad_ortho

        self.optimizer.step(closure)


    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        """Add a param group to the Optimizer."""
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        """String representation."""
        return f"OrthoGrad({self.optimizer.__repr__()})"

# Example usage for OrthoAdamW and OrthoSGD (you can create these if needed for clarity):
class OrthoAdamW(OrthoGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        base_optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(OrthoAdamW, self).__init__(base_optimizer)

class OrthoSGD(OrthoGrad):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        base_optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(OrthoSGD, self).__init__(base_optimizer)
