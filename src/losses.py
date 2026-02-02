import torch
import torch.nn.functional as F

def consistency_regularized_loss(outputs, kl_weight=0.1):
    """
    Computes the Marginal Entropy Loss + KL Divergence Penalty.
    outputs: Logits from B augmentations of the same image.
    """
    # 1. Marginal Entropy (Standard MEMO)
    avg_probs = F.softmax(outputs, dim=1).mean(dim=0)
    marginal_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    
    # 2. KL-Divergence Penalty (My Modification)
    # [Insert your specific KL-divergence logic here]
    
    return marginal_entropy + (kl_weight * kl_penalty)