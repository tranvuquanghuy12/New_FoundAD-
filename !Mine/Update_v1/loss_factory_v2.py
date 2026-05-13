import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OursTotalLoss_V2(nn.Module):
    """
    Update V1: Tích hợp Adaptive Margin và Prototypical Contrast.
    Phù hợp cho các bài báo chuẩn IEEE Transactions.
    """
    def __init__(self, loss_mode='l2', tau=0.1, lam1=1.0, lam2=1.0, lam3=1.0, lam4=1.0):
        super(OursTotalLoss_V2, self).__init__()
        self.loss_mode = loss_mode
        self.tau = tau
        self.lam1 = lam1 # Adaptive Margin weight
        self.lam2 = lam2 # Hard negative weight
        self.lam3 = lam3 # Graph weight
        self.lam4 = lam4 # Energy weight
        
        # Prototype để lưu trữ "linh hồn" của sự bình thường
        self.register_buffer("normal_prototype", None)

    def compute_adaptive_margin(self, h_normal):
        """
        Tính toán biên thích nghi dựa trên độ lệch chuẩn của các mẫu bình thường.
        """
        with torch.no_grad():
            # Khoảng cách trung bình giữa các mẫu bình thường
            dist_matrix = torch.cdist(h_normal.mean(dim=1), h_normal.mean(dim=1))
            std_dev = torch.std(dist_matrix)
            # Biên m = mean_dist + k * std_dev (k thường chọn là 1 hoặc 2)
            adaptive_m = dist_matrix.mean() + std_dev
            return adaptive_m.clamp(min=0.5, max=2.0)

    def prototypical_loss(self, h, p, is_anomaly, adaptive_m):
        """
        Ép các mẫu bình thường hội tụ về Prototype và đẩy mẫu lỗi ra xa Prototype.
        """
        # Cập nhật Prototype (Exponential Moving Average)
        h_normal = h[~is_anomaly].mean(dim=(0, 1)).detach() # [D]
        if self.normal_prototype is None:
            self.normal_prototype = h_normal
        else:
            self.normal_prototype = 0.9 * self.normal_prototype + 0.1 * h_normal

        # Tính khoảng cách tới Prototype
        dist_to_proto = torch.norm(p.mean(dim=1) - self.normal_prototype, p=2, dim=-1) # [B]
        
        loss_in = dist_to_proto[~is_anomaly].mean() if (~is_anomaly).any() else 0.0
        loss_out = F.relu(adaptive_m - dist_to_proto[is_anomaly]).mean() if is_anomaly.any() else 0.0
        
        return loss_in + loss_out

    def graph_consistency_v2(self, h, adj_matrix):
        """
        Graph Laplacian cải tiến: Bảo toàn cấu trúc Topo của Manifold.
        """
        B, N, D = h.shape
        h_flat = h.mean(dim=1) # [B, D]
        
        # Tính toán ma trận Laplacian L = D - A
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        laplacian = degree_matrix - adj_matrix
        
        # Trace(H.T * L * H)
        graph_loss = torch.trace(torch.mm(torch.mm(h_flat.t(), laplacian), h_flat)) / (B * B)
        return graph_loss

    def forward(self, h, p, is_anomaly=None, adj_matrix=None):
        # 1. Base Reconstruction Loss
        if self.loss_mode == 'l2':
            l_recon_base = F.mse_loss(h, p, reduction="none").mean(dim=(1, 2))
        else:
            l_recon_base = F.smooth_l1_loss(h, p, reduction="none").mean(dim=(1, 2))

        if is_anomaly is None:
            return {"total_loss": l_recon_base.mean()}

        # 2. Adaptive Margin & Prototypical Contrast
        # Chỉ dùng mẫu bình thường để tính biên
        m_adaptive = self.compute_adaptive_margin(h[~is_anomaly]) if (~is_anomaly).any() else torch.tensor(1.0).to(h.device)
        l_proto = self.prototypical_loss(h, p, is_anomaly, m_adaptive)

        # 3. Hard Negative Reweighting
        # Những mẫu lỗi có loss tái tạo thấp là những mẫu "hard"
        weights = torch.ones_like(l_recon_base)
        if is_anomaly.any():
            anomaly_mask = is_anomaly.bool()
            # Invert loss: low loss -> high weight
            weights[anomaly_mask] = 1.0 + torch.exp(-l_recon_base[anomaly_mask] / self.tau)
        l_recon = (l_recon_base * weights).mean()

        # 4. Graph Consistency V2
        l_graph = 0.0
        if adj_matrix is not None:
            l_graph = self.graph_consistency_v2(h, adj_matrix)

        # 5. Energy Loss (Giữ nguyên từ bản cũ nhưng dùng biên thích nghi)
        dist = torch.norm(h - p, p=2, dim=-1)
        energy = -self.tau * torch.log(torch.exp(-dist / self.tau).sum(dim=-1) + 1e-6)
        l_energy = energy[~is_anomaly].mean() if (~is_anomaly).any() else 0.0
        if is_anomaly.any():
            l_energy += F.relu(m_adaptive - energy[is_anomaly]).mean()

        # Tổng hợp
        total_loss = l_recon + self.lam1 * l_proto + self.lam3 * l_graph + self.lam4 * l_energy

        return {
            "total_loss": total_loss,
            "l_recon": l_recon,
            "l_proto": l_proto,
            "l_graph": l_graph,
            "l_energy": l_energy,
            "m_adaptive": m_adaptive
        }
