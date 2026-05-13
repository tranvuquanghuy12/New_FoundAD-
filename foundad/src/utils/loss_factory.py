import torch
import torch.nn as nn
import torch.nn.functional as F

class OursTotalLoss(nn.Module):
    def __init__(self, loss_mode='l2', margin=1.0, tau=0.1, lam1=1.0, lam3=1.0, lam4=0.01):
        super(OursTotalLoss, self).__init__()
        self.loss_mode = loss_mode
        self.margin = margin  # MUST be positive for relu to have effect
        self.tau = tau
        self.lam1 = lam1  # Anomaly Push weight
        self.lam3 = lam3  # Graph consistency weight
        self.lam4 = lam4  # Energy weight (regularizer)

    def reconstruction_loss(self, h, p):
        if self.loss_mode == 'l2':
            return F.mse_loss(h, p, reduction="none").mean(dim=-1)
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h, p, reduction="none").mean(dim=-1)
        return F.mse_loss(h, p, reduction="none").mean(dim=-1)

    def anomaly_push_loss(self, enc_context, p, is_anomaly, mask):
        """
        Self-Distance Anomaly Push Loss.

        Trực tiếp tối ưu metric test-time: đảm bảo điểm tái tạo của anomaly patch
        phải LỚN, bằng cách đo ||enc_abn - pred(enc_abn)|| thay vì ||h_clean - pred||.

        Điều này tránh hoàn toàn 'Score Inversion': nếu pred học identity mapping cho
        anomaly patches (pred ≈ enc_abn), loss sẽ phạt vì MSE(enc_abn, pred) → 0 < margin.

        Args:
            enc_context: Features trích xuất từ ảnh ANOMALOUS [B, N, D].
            p:           Output của predictor trên enc_context [B, N, D].
            is_anomaly:  Boolean mask [B], True = batch này có anomaly.
            mask:        Pixel-wise anomaly mask [B, 1, H, W].
        """
        if not is_anomaly.any() or mask is None:
            return torch.tensor(0.0, device=p.device)

        # MSE(enc_abn, pred) patch-wise [B, N]
        dist_patch = torch.mean((enc_context - p) ** 2, dim=-1)

        dist_anomaly = dist_patch[is_anomaly]     # [B_abn, N]
        mask_anomaly = mask[is_anomaly]            # [B_abn, 1, H, W]

        B_abn, N = dist_anomaly.shape
        feat_size = int(N ** 0.5)

        mask_patch = F.adaptive_avg_pool2d(mask_anomaly, (feat_size, feat_size))
        mask_patch = mask_patch.view(B_abn, -1)   # [B_abn, N]

        is_defect_patch = mask_patch > 0.5
        if not is_defect_patch.any():
            return torch.tensor(0.0, device=p.device)

        target_dists = dist_anomaly[is_defect_patch]
        # Hinge: penalize when MSE(enc_abn, pred) < margin (pred too close to enc_abn)
        loss_push = F.relu(self.margin - target_dists).mean()
        return loss_push

    def graph_consistency_loss(self, p, adj_matrix, is_anomaly):
        """
        Chỉ áp dụng cho mẫu Normal để giữ tính nhất quán đa tạp (Manifold Consistency).
        """
        mask_normal = ~is_anomaly
        if not mask_normal.any():
            return torch.tensor(0.0, device=p.device)

        p_normal = p[mask_normal].mean(dim=1)        # [B_normal, D]
        adj_normal = adj_matrix[mask_normal][:, mask_normal]

        dist_matrix = torch.cdist(p_normal, p_normal, p=2) ** 2  # [B_normal, B_normal]
        graph_loss = (adj_normal * dist_matrix).sum() / (p_normal.size(0) ** 2 + 1e-6)
        return graph_loss

    def energy_regularizer(self, h, p, is_anomaly):
        """
        Hinge Energy trên h (target clean) vs p.
        Chỉ dùng như regularizer nhẹ để đẩy distribution anomaly ra xa.
        Không phụ thuộc vào margin cùng chiều với anomaly_push.
        """
        if not is_anomaly.any():
            return torch.tensor(0.0, device=h.device)

        dist = torch.norm(h - p, p=2, dim=-1)  # [B, N]
        energy = -self.tau * torch.log(torch.exp(-dist / self.tau).sum(dim=-1) + 1e-6)  # [B]

        # Anomaly nên có energy CAO → penalize khi energy thấp
        loss_out = F.relu(self.margin - energy[is_anomaly]).mean()
        return loss_out

    def forward(self, h, p, is_anomaly=None, adj_matrix=None, mask=None, enc_context=None):
        """
        Hybrid Patch-level Loss (v2 - Self-Distance).

        Args:
            h:            Target features từ ảnh CLEAN [B, N, D].
            p:            Predictor output [B, N, D].
            is_anomaly:   Boolean [B], True khi batch là anomalous.
            adj_matrix:   Ma trận kề cho Graph Loss [B, B].
            mask:         Pixel-wise anomaly mask [B, 1, H, W].
            enc_context:  Features trích xuất từ ảnh ĐẦU VÀO (clean hoặc anomalous) [B, N, D].
                          Khi is_anomaly=True, đây là enc_abn — dùng cho Self-Distance Loss.
        """
        if is_anomaly is None:
            return self.reconstruction_loss(h, p).mean()

        B, N, D = h.shape

        # 1. Reconstruction Loss cấp PATCH (chỉ trên patch SẠCH, dùng h_clean vs p)
        recon_patch = self.reconstruction_loss(h, p)  # [B, N]
        H_feat = int(N ** 0.5)
        mask_down = F.adaptive_avg_pool2d(mask, (H_feat, H_feat)).view(B, N)
        is_clean_patch = (mask_down <= 0.5)
        l_recon = recon_patch[is_clean_patch].mean() if is_clean_patch.any() else torch.tensor(0.0, device=h.device)

        # 2. Self-Distance Anomaly Push Loss
        #    Dùng enc_context (= enc_abn khi anomaly) để tối ưu trực tiếp metric test-time.
        #    Nếu enc_context không được cung cấp, fallback về h (backward compatible).
        ctx = enc_context if enc_context is not None else h
        l_push = self.anomaly_push_loss(ctx, p, is_anomaly, mask)

        # 3. Graph Consistency (CHỈ normal)
        l_graph = torch.tensor(0.0, device=h.device)
        if adj_matrix is not None:
            l_graph = self.graph_consistency_loss(p, adj_matrix, is_anomaly)

        # 4. Energy Regularizer (nhẹ, chỉ anomaly)
        l_energy = self.energy_regularizer(h, p, is_anomaly)

        total_loss = l_recon + self.lam1 * l_push + self.lam3 * l_graph + self.lam4 * l_energy

        return {
            "total_loss": total_loss,
            "l_recon":    l_recon,
            "l_margin":   l_push,    # Giữ key "l_margin" để không cần sửa CSVLogger
            "l_graph":    l_graph,
            "l_energy":   l_energy,
        }

