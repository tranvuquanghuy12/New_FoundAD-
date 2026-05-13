import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import yaml
from torch.utils.data import DataLoader
from src.foundad import VisionModule
from src.datasets.dataset import TestDataset # Sử dụng TestDataset để có cả mẫu sạch và lỗi

def get_latents(model, dataloader, device, max_samples=200):
    model.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for i, (img, label, mask) in enumerate(dataloader):
            if i >= max_samples: break
            
            img = img.to(device)
            # Trích xuất đặc trưng từ encoder
            features = model.encoder(img) # [B, N, C]
            
            # Lấy vector trung bình của các patch (hoặc CLS token nếu có)
            latent = features.mean(dim=1) # [B, C]
            
            all_latents.append(latent.cpu().numpy())
            all_labels.append(label.numpy())
            
    return np.concatenate(all_latents), np.concatenate(all_labels)

def plot_tsne(latents, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    # Label 0 thường là 'good' (sạch), các label khác là lỗi
    indices_normal = where(labels == 0)
    indices_anomaly = where(labels != 0)
    
    plt.scatter(latents_2d[labels==0, 0], latents_2d[labels==0, 1], 
                c='blue', label='Normal (Clean)', alpha=0.6, edgecolors='w')
    plt.scatter(latents_2d[labels!=0, 0], latents_2d[labels!=0, 1], 
                c='red', label='Anomaly (Defect)', alpha=0.6, edgecolors='w')
    
    plt.title(title, fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"✅ Đã lưu biểu đồ tại: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Đường dẫn file .pth.tar")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file params.yaml")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="tsne_result.png")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Build Model
    mcfg = cfg["meta"]
    model = VisionModule(
        mcfg["model"], mcfg["pred_depth"], mcfg["pred_emb_dim"], 
        if_pe=mcfg.get("if_pred_pe", True), feat_normed=mcfg.get("feat_normed", False)
    ).to(device)
    
    # Load Weights
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # Build Dataset
    dataset = TestDataset(
        root=args.data_path,
        category=cfg["data"]["data_name"].replace("mvtec_", ""),
        input_size=mcfg["crop_size"]
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"🚀 Đang trích xuất đặc trưng từ {args.ckpt}...")
    latents, labels = get_latents(model, dataloader, device)
    
    print("🎨 Đang tính toán t-SNE (quá trình này có thể mất 1-2 phút)...")
    plot_tsne(latents, labels, f"Latent Space Visualization\n({args.save_name})", args.save_name)
