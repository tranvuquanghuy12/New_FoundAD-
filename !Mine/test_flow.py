import sys
import os
import torch
import torch.nn as nn

# Thêm thư mục gốc của dự án vào path để có thể import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from foundad.src.foundad import VisionModule
    from foundad.src.train import Trainer
    print("✅ Import các module thành công.")
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    sys.exit(1)

def test_initialization():
    print("\n--- Đang kiểm tra việc khởi tạo mô hình (Mock Backbone) ---")
    
    # Mock config cho VisionModule
    # Thay vì dùng 'dinov3' thật (sẽ gây tải file), ta sẽ mock hàm _build_encoder
    class MockVisionModule(VisionModule):
        def _build_encoder(self, model):
            print(f"   [Mock] Giả lập bộ mã hóa cho: {model}")
            # Giả lập encoder đơn giản để không phải tải weights
            enc = nn.Sequential(nn.Identity())
            num_patches = 256
            embed_dim = 768
            processor = None
            projector = None
            return enc, num_patches, embed_dim, processor, projector

    try:
        # Khởi tạo mô hình với predictor depth = 2, dim = 512
        model = MockVisionModule(
            model_name="mock_model", 
            pred_depth=2, 
            pred_emb_dim=512, 
            use_cuda=False
        )
        print("✅ Khởi tạo VisionModule thành công.")
        print(f"   - Predictor architecture: {type(model.predictor)}")
        print(f"   - Number of parameters in Predictor: {sum(p.numel() for p in model.predictor.parameters()):,}")
        
        # Test forward pass với dữ liệu giả
        dummy_input = torch.randn(1, 3, 224, 224)
        # Vì ta mock _extract nên cần mock cả nó hoặc gọi trực tiếp predictor
        z = torch.randn(1, 256, 768)
        p = model.predictor(z)
        print(f"✅ Kiểm tra forward pass của Predictor thành công. Output shape: {p.shape}")

    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo hoặc chạy thử: {e}")

if __name__ == "__main__":
    print("🚀 Bắt đầu script kiểm tra luồng chạy (không tải tài nguyên)...")
    test_initialization()
    print("\n🚀 Kiểm tra kết thúc.")
