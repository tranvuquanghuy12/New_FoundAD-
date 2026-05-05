# Kế hoạch Cải tiến Hàm Loss cho Dự án "Ours" (Based on FoundAD)

Tài liệu này trình bày chi tiết kế hoạch nâng cấp cơ chế phát hiện bất thường của FoundAD bằng cách thay thế hàm Loss MSE đơn giản bằng một hệ thống Loss đa thành phần, nhằm tối ưu hóa khả năng biểu diễn đặc trưng và độ chính xác trong kịch bản ít mẫu (Few-shot).

---

## 1. Bản chất của sự cải tiến (Nature)

Dự án **"Ours"** chuyển đổi tư duy từ việc chỉ "tái tạo đặc trưng" (Reconstruction) sang việc "quản lý không gian đa tạp" (Manifold Management). Thay vì chỉ ép mô hình khớp các điểm ảnh, chúng ta ép mô hình phải hiểu được mối quan hệ giữa các mẫu và ranh giới giữa cái bình thường và cái bất thường một cách tường minh.

## 2. Nguyên lý hoạt động (Principles)

Hàm Loss tổng quát mới sẽ được định nghĩa là:
**𝓛_total = 𝓛_recon + λ₁𝓛_margin + λ₂𝓛_hard + λ₃𝓛_graph + λ₄𝓛_energy**

### 2.1. Manifold Reconstruction (𝓛_recon)
*   **Nguyên lý:** Giữ nguyên cơ chế cốt lõi của FoundAD, sử dụng MSE hoặc Smooth L1 để chiếu đặc trưng về mặt đa tạp chuẩn.
*   **Mục tiêu:** Đảm bảo bộ chiếu (Predictor) vẫn giữ được khả năng "phục hồi" đặc trưng.

### 2.2. Margin-based Adaptive Similarity (𝓛_margin)
*   **Nguyên lý:** Sử dụng một khoảng cách biên (margin) để phân tách các vùng đặc trưng.
*   **Mục tiêu:** Ép các mẫu bất thường (synthetic anomalies) phải nằm cách xa vùng đa tạp bình thường ít nhất một khoảng **m**. Điều này ngăn chặn hiện tượng mô hình quá khớp (overfitting) và làm mờ ranh giới lỗi.

### 2.3. Hard Negative Reweighting (𝓛_hard)
*   **Nguyên lý:** Tự động tăng trọng số cho các mẫu lỗi giả lập mà mô hình đang tái tạo "quá tốt" (tức là những mẫu lỗi trông rất giống ảnh thường).
*   **Mục tiêu:** Ép mô hình phải học những chi tiết nhỏ nhất, tinh vi nhất thay vì chỉ học các lỗi lớn hiển nhiên.

### 2.4. Graph Consistency (𝓛_graph)
*   **Nguyên lý:** Xây dựng ma trận kề giữa các mẫu trong tập Few-shot.
*   **Mục tiêu:** Ràng buộc bộ chiếu phải bảo toàn cấu trúc liên kết giữa các mẫu. Nếu mẫu A và B bình thường giống nhau, thì sau khi qua bộ chiếu chúng vẫn phải giống nhau. Điều này ổn định hóa bề mặt đa tạp.

### 2.5. Energy-based Open-world Loss (𝓛_energy)
*   **Nguyên lý:** Áp dụng hàm năng lượng để gán mức năng lượng thấp cho vùng bình thường và cao cho vùng bất thường.
*   **Mục tiêu:** Cung cấp một cơ chế ra quyết định (scoring) tốt hơn Distance Map truyền thống, đặc biệt hiệu quả trong việc phát hiện các loại lỗi chưa từng thấy trong quá trình huấn luyện (Open-world).

---

## 3. Sự khác biệt so với FoundAD gốc (Differences)

| Đặc điểm | FoundAD (Original) | Dự án "Ours" (Improved) |
| :--- | :--- | :--- |
| **Hàm Loss** | Đơn mục tiêu (MSE) | Đa mục tiêu (Hybrid Loss) |
| **Quan hệ mẫu** | Độc lập (Per-patch) | Liên kết (Graph-aware) |
| **Xử lý mẫu lỗi** | Coi mọi lỗi giả lập như nhau | Tập trung vào lỗi khó (Hard Negative) |
| **Độ phân tách** | Không có biên (No margin) | Có biên thích nghi (Adaptive Margin) |
| **Cơ chế Scoring** | Distance Map thuần túy | Kết hợp Energy Score |

---

## 4. Lộ trình triển khai (Roadmap)

### Bước 1: Cấu trúc lại Module Loss
*   Tạo file `src/utils/loss_factory.py` để định nghĩa các thành phần loss mới.
*   Tích hợp hàm tính ma trận kề (Adjacency Matrix) vào quá trình nạp dữ liệu.

### Bước 2: Sửa đổi Trainer (`src/train.py`)
*   Thay thế hàm `_loss_fn` bằng lớp `OursTotalLoss`.
*   Bổ sung cơ chế điều phối trọng số λ (Lambda) theo epoch để ổn định quá trình hội tụ.

### Bước 3: Nâng cấp bộ suy luận (`src/AD.py`)
*   Bổ sung hàm tính Energy Score.
*   Kết hợp Distance Map và Energy Map để tạo ra Heatmap cuối cùng có độ nhiễu thấp hơn.

### Bước 4: Thử nghiệm và Căn chỉnh
*   Huấn luyện trên tập dữ liệu chuẩn (MVTec-AD) để kiểm tra tính ổn định.
*   So sánh kết quả AUC và PRO với phiên bản FoundAD gốc.

---
*Tài liệu được khởi tạo ngày 03/05/2026 cho dự án nghiên cứu cải tiến Visual Anomaly Detection.*
