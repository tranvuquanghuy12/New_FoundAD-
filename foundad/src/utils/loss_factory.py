import torch
import torch.nn as nn
import torch.nn.functional as F

class OursTotalLoss(nn.Module):
    def __init__(self, loss_mode='l2', margin=1.0, tau=0.1, lam1=1.0, lam2=1.0, lam3=1.0, lam4=1.0):
        super(OursTotalLoss, self).__init__()
        self.loss_mode = loss_mode
        self.margin = margin
        self.tau = tau
        self.lam1 = lam1 # Margin weight
        self.lam2 = lam2 # Hard negative weight
        self.lam3 = lam3 # Graph weight
        self.lam4 = lam4 # Energy weight

    def reconstruction_loss(self, h, p):
        if self.loss_mode == 'l2':
            return F.mse_loss(h, p, reduction="none").mean(dim=-1)
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h, p, reduction="none").mean(dim=-1)
        return F.mse_loss(h, p, reduction="none").mean(dim=-1)

    def margin_loss(self, h, p, is_anomaly):
        """
        is_anomaly: boolean tensor [B]
        """
        dist = torch.norm(h - p, p=2, dim=-1)
        # Normal samples should have dist -> 0
        # Anomalous samples should have dist > margin
        loss_normal = dist[~is_anomaly].mean() if (~is_anomaly).any() else 0.0
        loss_anomaly = F.relu(self.margin - dist[is_anomaly]).mean() if is_anomaly.any() else 0.0
        return loss_normal + loss_anomaly

    def hard_negative_reweighting(self, individual_losses, is_anomaly):
        """
        individual_losses: [B, N] or [B]
        """
        if not is_anomaly.any():
            return individual_losses.mean()
        
        # Focus on anomalous samples that have low reconstruction loss (hard to detect)
        weights = torch.ones_like(individual_losses)
        # Higher weights for anomalies with lower loss
        anomaly_indices = torch.where(is_anomaly)[0]
        if len(anomaly_indices) > 0:
            # Simple inverse weighting for hard samples
            norm_loss = individual_losses[anomaly_indices] / (individual_losses[anomaly_indices].max() + 1e-6)
            weights[anomaly_indices] = 1.0 + (1.0 - norm_loss)
            
        return (individual_losses * weights).mean()

    def graph_consistency_loss(self, h, adj_matrix):
        """
        adj_matrix: [B, B] precomputed similarity matrix
        h: [B, N, D] features
        """
        # Flatten patches if needed or compute per sample
        # L_graph = sum Aij * ||hi - hj||^2
        B, N, D = h.shape
        h_mean = h.mean(dim=1) # [B, D] average patch features per image
        
        dist_matrix = torch.cdist(h_mean, h_mean, p=2)**2 # [B, B]
        graph_loss = (adj_matrix * dist_matrix).sum() / (B * B)
        return graph_loss

    def energy_loss(self, h, p, is_anomaly):
        """
        Energy E(x) = -tau * log(sum(exp(-d/tau)))
        """
        dist = torch.norm(h - p, p=2, dim=-1) # [B, N]
        energy = -self.tau * torch.log(torch.exp(-dist / self.tau).sum(dim=-1) + 1e-6) # [B]
        
        # Push energy of normal samples down, anomaly samples up
        loss_in = energy[~is_anomaly].mean() if (~is_anomaly).any() else 0.0
        loss_out = F.relu(self.margin - energy[is_anomaly]).mean() if is_anomaly.any() else 0.0
        return loss_in + loss_out

    def forward(self, h, p, is_anomaly=None, adj_matrix=None):
        """
        h: target features [B, N, D]
        p: predicted features [B, N, D]
        is_anomaly: [B] boolean
        """
        # 1. Base Reconstruction
        recon_ind = self.reconstruction_loss(h, p) # [B, N]
        
        if is_anomaly is None:
            return recon_ind.mean()
            
        # 2. Hard Negative Reweighting
        l_recon = self.hard_negative_reweighting(recon_ind.mean(dim=-1), is_anomaly)
        
        # 3. Margin Loss
        l_margin = self.margin_loss(h, p, is_anomaly)
        
        # 4. Graph Consistency
        l_graph = 0.0
        if adj_matrix is not None:
            l_graph = self.graph_consistency_loss(h, adj_matrix)
            
        # 5. Energy Loss
        l_energy = self.energy_loss(h, p, is_anomaly)
        
        total_loss = l_recon + self.lam1 * l_margin + self.lam3 * l_graph + self.lam4 * l_energy
        
        return {
            "total_loss": total_loss,
            "l_recon": l_recon,
            "l_margin": l_margin,
            "l_graph": l_graph,
            "l_energy": l_energy
        }
