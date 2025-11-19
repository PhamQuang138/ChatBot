CHÚ Ý: ĐỂ SỬ DỤNG CẦN LẤY ACCESS TOKEN CỦA LLAMA 3.2 1B và chú ý cài các phiên bản thư viện phù hợp cho Windows (nếu không dùng ubuntu)

⚖️ Trợ lý Pháp lý Luật Dược Việt Nam (RAG Chatbot)
Hệ thống Chatbot hỗ trợ tra cứu và hỏi đáp về Luật Dược Việt Nam, được xây dựng dựa trên kiến trúc RAG (Retrieval-Augmented Generation). Hệ thống kết hợp khả năng tìm kiếm ngữ nghĩa (Semantic Search) và tìm kiếm từ khóa (Keyword Search) để đưa ra câu trả lời chính xác, có trích dẫn nguồn cụ thể.

🚀 Tính năng nổi bật
Tìm kiếm lai (Hybrid Search): Kết hợp BM25 và Vector Search (ChromaDB) để tối ưu hóa kết quả truy xuất.

Hai chế độ hoạt động (Dual-Mode):

Chế độ Từ điển (Retrieval-Only): Trả về văn bản gốc tức thì khi tra cứu điều luật cụ thể (VD: "Điều 47").

Chế độ Chatbot (Generative): Sử dụng LLM để tổng hợp, giải thích và tư vấn các câu hỏi phức tạp.

Đánh giá thời gian thực: Hiển thị các chỉ số Context Relevance, Groundedness, và Answer Relevance ngay trên giao diện.

Trích dẫn nguồn minh bạch: Luôn hiển thị Điều luật/Văn bản gốc đi kèm câu trả lời.

🛠️ Kiến trúc kỹ thuật
LLM: meta-llama/Llama-3.2-1B-Instruct (Quantized 4-bit).

LoRA Adapter: Tinh chỉnh trên dữ liệu Luật Dược.

Embedding Model: Qwen/Qwen3-Embedding-0.6B.

Vector Database: ChromaDB.

Interface: Gradio.

📋 Yêu cầu hệ thống (Prerequisites)
Hệ điều hành: Linux (Ubuntu) hoặc Windows.

Python: Phiên bản 3.10 trở lên.

Phần cứng:

GPU (Khuyến nghị): NVIDIA GPU với tối thiểu 4GB VRAM (để chạy mượt mà Llama 1B + Embedding).

CPU: Có thể chạy nhưng tốc độ phản hồi sẽ chậm.

RAM: Tối thiểu 16GB.

📦 Hướng dẫn Cài đặt (Installation)
Làm theo các bước sau để thiết lập môi trường và chạy dự án.

Bước 1: Clone dự án
Tải mã nguồn về máy của bạn:

Bash

git clone https://github.com/PhamQuang138/ChatBot

Bước 2: Tạo môi trường ảo (Virtual Environment)
Sử dụng venv để tạo môi trường độc lập, tránh xung đột thư viện:

Trên Windows:

Bash

python -m venv venv
Trên Linux/macOS:

Bash

python3 -m venv venv
Bước 3: Kích hoạt môi trường ảo
Trên Windows (Command Prompt):

DOS

venv\Scripts\activate
(Hoặc PowerShell: venv\Scripts\Activate.ps1)

Trên Linux/macOS:

Bash

source venv/bin/activate
(Sau khi kích hoạt, bạn sẽ thấy chữ (venv) xuất hiện ở đầu dòng lệnh)

Bước 4: Cài đặt các thư viện phụ thuộc
Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết từ file requirements.txt:

Bash

pip install --upgrade pip

pip install -r requirements.txt

(Lưu ý: Nếu bạn dùng GPU NVIDIA, hãy đảm bảo đã cài đặt PyTorch bản hỗ trợ CUDA tương thích).

📂 Cấu trúc dữ liệu
Đảm bảo bạn đã đặt dữ liệu và model vào đúng thư mục trước khi chạy:

Plaintext
Chatbot/

├── data/
  
│  └── chroma_db_qwen_embed_vn/  # Thư mục chứa Vector Database

├── src/

│  └── lora_llama3_4bit/         # Thư mục chứa Adapter LoRA (nếu có)

├── rag_qwen4b_gradio.py          # File code chính

├── requirements.txt              # Danh sách thư viện

└── README.md

Lưu ý: Trong file rag_qwen4b_gradio.py, hãy kiểm tra biến BASE_DIR để đảm bảo đường dẫn trỏ đúng tới thư mục dự án của bạn.

▶️ Hướng dẫn Sử dụng
Chạy ứng dụng:

Bash

python rag_qwen4b_gradio.py
Truy cập giao diện: Mở trình duyệt và truy cập địa chỉ (thường là): http://localhost:7860

Thao tác:

Nhập câu hỏi vào ô trống.

Tick chọn "Gọi LLM" nếu muốn Chatbot trả lời chi tiết.

Bỏ chọn "Gọi LLM" nếu chỉ muốn tìm kiếm văn bản gốc nhanh chóng.

📊 Giải thích các chỉ số đánh giá (Metrics)

Context Relevance: Đo độ liên quan giữa Câu hỏi và Văn bản luật tìm được.

Groundedness: Đo độ trung thực, xem Câu trả lời của AI có bám sát Văn bản luật không (chống bịa đặt).

Answer Relevance: Đo xem Câu trả lời có đi đúng trọng tâm Câu hỏi không.




