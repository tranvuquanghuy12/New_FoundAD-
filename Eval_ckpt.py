import os
import sys
import subprocess
import re

def main():
    print("="*60)
    print("🚀 CÔNG CỤ ĐÁNH GIÁ CHECKPOINT FOUNDAD (AUROC/PRO-AUC) 🚀")
    print("="*60)
    
    ckpt_path = input("Nhập đường dẫn tới file weights (.pth.tar) bạn muốn đánh giá: ").strip()
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy file: {ckpt_path}")
        return

    # Chuẩn hóa đường dẫn
    ckpt_path = os.path.abspath(ckpt_path)
    parts = ckpt_path.split(os.sep)
    
    # Giả sử cấu trúc: .../logs/<run_name>/<variant>/<class_name>/train-epoch<N>.pth.tar
    # Tìm index của folder 'logs'
    try:
        logs_idx = parts.index('logs')
        run_name = parts[logs_idx + 1]
        variant = parts[logs_idx + 2]
        class_name = parts[logs_idx + 3]
        file_name = parts[-1]
    except (ValueError, IndexError):
        print("❌ Cấu trúc đường dẫn không hợp lệ. Phải nằm trong folder logs/RUN_NAME/VARIANT/CLASS/")
        return

    # Trích xuất epoch từ file name (ví dụ: train-epoch50.pth.tar -> 50)
    match = re.search(r'epoch(\d+)', file_name)
    if match:
        ckpt_epoch = match.group(1)
    else:
        ckpt_epoch = "50" # Mặc định
        
    print(f"📦 Thông tin trích xuất:")
    print(f"   - Run Name: {run_name}")
    print(f"   - Variant : {variant}")
    print(f"   - Class   : {class_name}")
    print(f"   - Epoch   : {ckpt_epoch}")
    print("-" * 30)

    # Đường dẫn dữ liệu (Anh có thể sửa lại nếu data nằm chỗ khác)
    data_path = "/home/ptitdemo/TVQH_TLU/Research Project 2/Dataset/MVTec AD"
    
    # Xây dựng câu lệnh
    cmd = [
        "python", "foundad/main.py",
        "mode=AD",
        "app=train_dinov3",
        f"data.data_name={class_name}",
        "data.dataset=mvtec",
        f"data.data_path={data_path}",
        f"data.test_root={data_path}",
        f"+run_name={run_name}",
        f"+variant={variant}",
        f"+ckpt_epoch={ckpt_epoch}"
    ]

    print(f"🏃 Đang thực thi lệnh đánh giá...")
    print(f"CMD: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("✅ HOÀN TẤT ĐÁNH GIÁ!")
        # Chỉ ra nơi lưu kết quả
        eval_folder = os.path.join(os.path.dirname(ckpt_path), f"eval/{ckpt_epoch}")
        print(f"📊 Kết quả chi tiết đã được lưu tại: {eval_folder}")
        print("="*60)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi thực thi lệnh AD: {e}")

if __name__ == "__main__":
    main()
