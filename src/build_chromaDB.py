from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch, os, json, shutil
import numpy as np

# ============ CONFIG ============
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "law_pharmacy_en_clean.json")
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db_qwen_embed")
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
# =================================

# 🔹 Load embedding model Qwen3-Embedding-0.6B
print(f"🛠️ Loading embedding model: {EMBED_MODEL} ...")

tokenizer_embed = AutoTokenizer.from_pretrained(EMBED_MODEL)
model_embed = AutoModel.from_pretrained(
    EMBED_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ Embedding model loaded successfully.")


# ===== Class Embedding cho LangChain =====
class Qwen3Embedding(Embeddings):
    def __init__(self, model, tokenizer, device="cpu", batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model_name = EMBED_MODEL

        if device.startswith("cuda"):
            self.total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        else:
            self.total_vram_gb = None

    def _print_vram_usage(self, batch_index):
        if self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(
                f"⚡ Batch {batch_index}: VRAM Alloc {allocated:.2f} GB / Reserved {reserved:.2f} GB / Total {self.total_vram_gb:.2f} GB"
            )

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state[:, 0, :]  # CLS token
                batch_emb = last_hidden / last_hidden.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.cpu().numpy())

            self._print_vram_usage(i // self.batch_size + 1)

        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ===== Build Vector DB =====
def build_vector_db(force_rebuild=False):
    if force_rebuild and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Đã xoá vector DB cũ. Bắt đầu xây dựng lại.")

    print("📄 Đang load dữ liệu JSON...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ File JSON không đúng định dạng: phải là list các object chứa article & content")

    docs = [
        Document(
            page_content=f"Article {d.get('article', '')}\n{d.get('content', '')}",
            metadata={"article": d.get("article", "")}
        )
        for d in data if d.get("content")
    ]

    print(f"📚 Tổng số documents: {len(docs)}")

    embedding_fn = Qwen3Embedding(model_embed, tokenizer_embed, device=DEVICE, batch_size=BATCH_SIZE)
    # Test embedding sanity check
    test_vec = embedding_fn.embed_query("Hello world")
    print(f"🔍 Test embedding shape: {np.array(test_vec).shape}")

    print(f"✨ Đang tạo mới Chroma DB tại {CHROMA_PATH} ...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,  # ✅ Đúng cú pháp
        persist_directory=CHROMA_PATH

    )

    vectordb.persist()
    print(f"✅ Vector DB đã lưu thành công tại: {CHROMA_PATH}")

    # Kiểm tra dimension
    try:
        test_emb = embedding_fn.embed_query("dimension check")
        print(f"✅ Kiểm tra dimension: {len(test_emb)}D — khớp với Qwen3-Embedding-0.6B")
    except Exception as e:
        print(f"⚠️ Không thể kiểm tra dimension: {e}")


if __name__ == "__main__":
    build_vector_db(force_rebuild=True)
