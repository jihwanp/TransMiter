import torch

def logit_stand(logits, return_others=False):
    """Standardize the logits by subtracting the mean and dividing by the standard deviation."""
    logits = logits
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    if return_others:
    
        return (logits - mean) / (std+1e-6), mean, std
    else:
        return (logits - mean) / (std+1e-6)