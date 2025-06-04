import torch

def compute_token_entropy(logits):
    """Compute average token entropy for uncertainty estimation."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.mean().item()