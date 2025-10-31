import re
import json
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "law.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "law_pharmacy_en_clean.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# 1️⃣ Loại bỏ các dòng không cần thiết
text = re.sub(r"Unofficial translation.*?\n", "", text)
text = re.sub(r"www\.economica\.vn", "", text)
text = re.sub(r"–|—", "-", text)
text = re.sub(r"\s+", " ", text)

# 2️⃣ Tách các điều luật
# Ví dụ: "Article 1. Scope..." => nhóm (Article 1. Scope...) và nội dung tiếp theo
pattern = r"(Article\s+\d+\..*?)(?=Article\s+\d+\.|$)"
articles = re.findall(pattern, text, re.DOTALL)

# 3️⃣ Chuyển thành danh sách dict
data = []
for article_text in articles:
    lines = article_text.strip().split(" ", 2)
    # Lấy tiêu đề điều
    title_match = re.match(r"(Article\s+\d+\.\s+)(.*)", article_text)
    if title_match:
        article_title = title_match.group(1).strip()
        content = title_match.group(2).strip()
        data.append({"article": article_title, "content": content})

# 4️⃣ Xuất JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned {len(data)} articles saved to {OUTPUT_FILE}")
