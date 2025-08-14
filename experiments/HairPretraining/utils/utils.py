import torch
import random
import numpy as np

def get_optimizer(model, lr, weight_decay, beta1, beta2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.endswith(".bias") or "bn" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return torch.optim.Adam(param_groups, lr, betas=(beta1, beta2))

def linear_increase_alpha(start_alpha, current_epoch, max_epochs, alpha_max=0.9):
    alpha = start_alpha + (alpha_max - start_alpha) * (current_epoch / max_epochs)
    return min(alpha_max, alpha)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

