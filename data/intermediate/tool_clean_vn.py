import os
import re
import json

# ===============================
# 1️⃣ Đường dẫn
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "luat_vn.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "law_chunks.json")

# ===============================
# 2️⃣ Hàm làm sạch văn bản
# ===============================
def clean_text(text):
    text = re.sub(r"–|—", "-", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Xóa số trang nằm giữa dòng
    text = re.sub(r"(\d+)\s+(?=(Điều|Điều)\s+\d+\.)", "", text)
    # Chuẩn hóa xuống dòng trước "Điều" hoặc "Chương"
    text = re.sub(r"(?<!\n)(?=(?:Điều|Điều|Chương)\s+\d+)", "\n", text)
    return text.strip()

# ===============================
# 3️⃣ Hàm tách Điều (dừng khi gặp “Điều” hoặc “Chương” kế tiếp)
# ===============================
def split_articles(text):
    pattern = r"((?:Điều|Điều)\s+\d+\..*?)(?=(?:\n(?:Điều|Điều|Chương)\s+\w+|$))"
    articles = re.findall(pattern, text, flags=re.DOTALL)

    cleaned = []
    for a in articles:
        # Cắt phần “Chương ...” nếu nó nằm sau phần cuối của điều
        a = re.split(r"\n?Chương\s+[IVXLC\d]+\s+", a, maxsplit=1, flags=re.IGNORECASE)[0]
        a = a.strip()
        if len(a) > 30:
            cleaned.append(a)
    return cleaned

# ===============================
# 4️⃣ Tiền xử lý chính (KHÔNG CHIA CHUNK)
# ===============================
def preprocess_law():
    print("📖 Đang xử lý file luật...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    articles = split_articles(text)
    print(f"🔍 Đã phát hiện {len(articles)} điều luật.")

    data = []
    for article_text in articles:
        match = re.match(r"(Điều|Điều)\s+(\d+)\.\s*(.*)", article_text, re.DOTALL)
        if not match:
            continue

        article_num = match.group(2).strip()
        article_title = f"Điều {article_num}."
        article_content = match.group(3).strip()

        # Không chia nhỏ — mỗi Điều là một phần tử duy nhất
        data.append({
            "article": article_title,
            "content": article_content
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo {len(data)} điều luật, lưu tại: {OUTPUT_FILE}")

# ===============================
# 5️⃣ Chạy trực tiếp
# ===============================
if __name__ == "__main__":
    preprocess_law()
