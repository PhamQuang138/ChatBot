import json
import re

# === ĐƯỜNG DẪN FILE JSON ===
INPUT_FILE = "QA_split.json"
OUTPUT_FILE = "conversations_clean.json"

def remove_asterisks_from_list_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ File JSON không phải dạng list. Hãy kiểm tra lại.")
        return

    for item in data:
        if "conversations" in item and isinstance(item["conversations"], list):
            for convo in item["conversations"]:
                if "value" in convo and isinstance(convo["value"], str):
                    convo["value"] = re.sub(r"\*\*", "", convo["value"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã xoá ** trong tất cả các hội thoại và lưu vào: {output_file}")

# === CHẠY ===
remove_asterisks_from_list_json(INPUT_FILE, OUTPUT_FILE)
