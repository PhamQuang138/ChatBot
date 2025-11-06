import json

INPUT_FILE = "luat_vn (1).json"
OUTPUT_FILE = "luat_vn_split.json"

def split_conversations(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    # Trường hợp file là list các hội thoại lớn
    if isinstance(data, list):
        for item in data:
            convos = item.get("conversations", [])
            # Duyệt từng cặp user–assistant
            for i in range(0, len(convos) - 1, 2):
                if convos[i]["from"] == "user" and convos[i + 1]["from"] == "assistant":
                    new_data.append({
                        "conversations": [convos[i], convos[i + 1]]
                    })

    # Trường hợp file là 1 object chứa key "conversations"
    elif isinstance(data, dict) and "conversations" in data:
        convos = data["conversations"]
        for i in range(0, len(convos) - 1, 2):
            if convos[i]["from"] == "user" and convos[i + 1]["from"] == "assistant":
                new_data.append({
                    "conversations": [convos[i], convos[i + 1]]
                })

    else:
        print("❌ File không có cấu trúc hợp lệ (không tìm thấy 'conversations').")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tách thành {len(new_data)} đoạn hội thoại và lưu vào: {output_file}")

if __name__ == "__main__":
    split_conversations(INPUT_FILE, OUTPUT_FILE)
