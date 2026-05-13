import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader

# Add root directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'foundad')))

from src.foundad import VisionModule
from src.datasets.dataset import TestDataset

def main():
    print("="*60)
    print("📊 CÔNG CỤ PHÂN TÍCH FOUNDAD (CONVERGENCE & T-SNE) 📊")
    print("="*60)
    ckpt_path = input("Nhập đường dẫn tới file weights (.pth.tar) bạn muốn phân tích: ").strip()
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy file checkpoint: {ckpt_path}")
        sys.exit(1)
        
    log_dir = os.path.dirname(ckpt_path)
    csv_path = os.path.join(log_dir, "train.csv")
    yaml_path = os.path.join(log_dir, "params.yaml")
    
    # ---------------------------------------------------------
    # 1. Biểu đồ hội tụ (Convergence Chart)
    # ---------------------------------------------------------
    print("\n[1/2] Đang vẽ Biểu đồ hội tụ (Convergence Chart)...")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        is_baseline = False
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                cfg_temp = yaml.safe_load(f)
                is_baseline = (cfg_temp.get("variant") == "Baseline")

        plt.figure(figsize=(12, 6))
        if is_baseline:
            if 'loss_total' in df.columns: plt.plot(df['epoch'], df['loss_total'], label='Total Loss (Reconstruction)', linewidth=2, color='blue')
            plt.title('Detailed Convergence Analysis - Baseline', fontsize=16)
        else:
            if 'loss_total' in df.columns: plt.plot(df['epoch'], df['loss_total'], label='Total Loss', linewidth=2, color='black')
            if 'l_recon' in df.columns: plt.plot(df['epoch'], df['l_recon'], label='Reconstruction', alpha=0.7)
            if 'l_margin' in df.columns: plt.plot(df['epoch'], df['l_margin'], label='Margin', alpha=0.7)
            if 'l_graph' in df.columns: plt.plot(df['epoch'], df['l_graph'], label='Graph', alpha=0.7)
            if 'l_energy' in df.columns: plt.plot(df['epoch'], df['l_energy'], label='Energy', alpha=0.7)
            plt.title('Detailed Convergence Analysis - Hybrid Loss (Ours)', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.yscale('linear')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # Cắt ngọn Y-axis để nhìn rõ các đường Loss nhỏ giống Baseline
        if not is_baseline:
            plt.ylim(0, 0.005)
            
        plt.legend()
        conv_save_path = os.path.join(log_dir, "convergence_chart.png")
        plt.savefig(conv_save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu Biểu đồ hội tụ tại: {conv_save_path}")
    else:
        print(f"⚠️ Không tìm thấy file train.csv tại {csv_path}. Bỏ qua vẽ Convergence.")

    # ---------------------------------------------------------
    # 2. Biểu đồ không gian ẩn (t-SNE)
    # ---------------------------------------------------------
    print("\n[2/2] Đang vẽ Biểu đồ không gian ẩn (t-SNE)...")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mcfg = cfg["meta"]
        
        model = VisionModule(
            mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], 
            if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False)
        ).to(device)
        
        print("📥 Đang load weights...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "predictor" in checkpoint: # Format mới
            model.predictor.load_state_dict(checkpoint["predictor"])
            if checkpoint.get("projector") and model.projector:
                model.projector.load_state_dict(checkpoint["projector"])
        elif "model" in checkpoint: # Format cũ
            model.load_state_dict(checkpoint["model"], strict=False)
            
        model.eval()
        
        print("📂 Đang load dữ liệu test...")
        data_name = cfg["data"]["data_name"].replace("mvtec_", "").replace("visa_", "")
        test_root = cfg["data"].get("test_root", f"/kaggle/input/datasets/ipythonx/mvtec-ad")
        
        test_dataset = TestDataset(source=test_root, classname=data_name, resize=mcfg.get("crop_size", 512))
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)
        
        all_latents, all_labels = [], []
        print("🧠 Đang trích xuất đặc trưng...")
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(all_latents) > 300: break # Giới hạn mẫu
                imgs = batch["image"].to(device)
                labels = batch["is_anomaly"].to(device)
                
                dummy_paths = [""] * imgs.size(0)
                h, p = model.context_features(imgs, dummy_paths, n_layer=mcfg.get("n_layer", 3), use_tensor_feat=True)
                
                # Tính Error Vector (h - p)
                error = h - p  # [B, N, D]
                
                # Tính độ lớn của Error (giống hệt cách AD.py tính Anomaly Score)
                error_magnitude = (error ** 2).mean(dim=-1)  # [B, N]
                
                # Lấy Top 10% patch có Error lớn nhất (vì Anomaly chỉ chiếm diện tích nhỏ)
                K_val = max(1, error.size(1) // 10)
                _, topk_idx = torch.topk(error_magnitude, k=K_val, dim=1) # [B, K]
                
                # Trích xuất các Error Vector của Top K patch này
                B, N, D = error.shape
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, K_val)
                topk_errors = error[batch_indices, topk_idx, :] # [B, K, D]
                
                # Lấy trung bình Error trên tập Top-K để đại diện cho ảnh
                latent = topk_errors.mean(dim=1).cpu().numpy() # [B, D]
                
                all_latents.append(latent)
                all_labels.append(labels.cpu().numpy())

        if len(all_latents) > 0:
            latents = np.concatenate(all_latents)
            labels = np.concatenate(all_labels)
            
            print("🎨 Đang chạy thuật toán t-SNE (1-2 phút)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latents_2d = tsne.fit_transform(latents)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(latents_2d[labels==0, 0], latents_2d[labels==0, 1], 
                        c='blue', label='Normal (Clean)', alpha=0.6, edgecolors='w', s=50)
            plt.scatter(latents_2d[labels!=0, 0], latents_2d[labels!=0, 1], 
                        c='red', label='Anomaly (Defect)', alpha=0.6, edgecolors='w', s=50)
            
            if is_baseline:
                plt.title('Latent Space Visualization (t-SNE)\nBaseline - Reconstruction Only', fontsize=15)
            else:
                plt.title('Latent Space Visualization (t-SNE)\nOurs - Hybrid Loss', fontsize=15)
            plt.legend()
            plt.grid(True, alpha=0.3)
            tsne_save_path = os.path.join(log_dir, "tsne_analysis.png")
            plt.savefig(tsne_save_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Đã lưu Biểu đồ t-SNE tại: {tsne_save_path}")
        else:
            print("⚠️ Không lấy được đặc trưng nào để vẽ t-SNE.")
            
    else:
        print(f"❌ Không tìm thấy params.yaml tại {yaml_path}. Không thể vẽ t-SNE.")
        
    print("\n🎉 HOÀN TẤT PHÂN TÍCH!")

if __name__ == "__main__":
    main()
