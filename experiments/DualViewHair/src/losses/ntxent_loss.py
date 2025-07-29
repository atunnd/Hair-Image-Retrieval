"""
Simple NT-Xent loss implementation for DualViewHair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Simple NT-Xent loss for dual-view contrastive learning.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_proj, teacher_proj):
        """
        Compute bidirectional NT-Xent loss.
        
        Args:
            student_proj: [B, D] student projections
            teacher_proj: [B, D] teacher projections
            
        Returns:
            NT-Xent loss scalar
        """
        batch_size = student_proj.size(0)
        
        # Normalize
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Concatenate: [2B, D]
        features = torch.cat([student_proj, teacher_proj], dim=0)
        
        # Similarity matrix: [2B, 2B]
        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Positive pairs:
        # student_i -> teacher_i (index i -> i+B)
        # teacher_i -> student_i (index i+B -> i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # [B, 2B-1, 2B-2, ...]
            torch.arange(0, batch_size)                # [0, 1, 2, ..., B-1]
        ]).to(sim_matrix.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class AsymmetricNTXentLoss(nn.Module):
    """
    Asymmetric NT-Xent loss with different weights for each direction.
    """
    
    def __init__(self, temperature=0.07, student_weight=1.0, teacher_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.student_weight = student_weight
        self.teacher_weight = teacher_weight
    
    def forward(self, student_proj, teacher_proj):
        """
        Compute weighted bidirectional NT-Xent loss.
        
        Args:
            student_proj: [B, D] student projections
            teacher_proj: [B, D] teacher projections
            
        Returns:
            Weighted NT-Xent loss scalar
        """
        batch_size = student_proj.size(0)
        
        # Normalize
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Student -> Teacher loss
        s2t_sim = torch.matmul(student_proj, teacher_proj.t()) / self.temperature
        s2t_labels = torch.arange(batch_size, device=s2t_sim.device)
        s2t_loss = F.cross_entropy(s2t_sim, s2t_labels)
        
        # Teacher -> Student loss
        t2s_sim = torch.matmul(teacher_proj, student_proj.t()) / self.temperature
        t2s_labels = torch.arange(batch_size, device=t2s_sim.device)
        t2s_loss = F.cross_entropy(t2s_sim, t2s_labels)
        
        # Weighted combination
        total_loss = self.student_weight * s2t_loss + self.teacher_weight * t2s_loss
        
        return total_loss
