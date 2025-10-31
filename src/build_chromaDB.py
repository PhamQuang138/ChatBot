from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch, os, json, shutil
import numpy as np

# ============ CONFIG ============
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "law_chunks.json")
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db_qwen_embed_vn")
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
# =================================

print(f"🛠️ Loading embedding model: {EMBED_MODEL} ...")
tokenizer_embed = AutoTokenizer.from_pretrained(EMBED_MODEL)
model_embed = AutoModel.from_pretrained(
    EMBED_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ Embedding model loaded successfully.")


# ===== Class Embedding =====
class Qwen3Embedding(Embeddings):
    def __init__(self, model, tokenizer, device="cpu", batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state[:, 0, :]
                batch_emb = last_hidden / last_hidden.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.cpu().numpy())
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ===== Build Vector DB =====
def build_vector_db(force_rebuild=False):
    if force_rebuild and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Đã xoá vector DB cũ.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        Document(
            page_content=f"{d.get('article', '').strip()}\n{d.get('content', '').strip()}",
            metadata={"article": d.get("article", "").strip()}
        )
        for d in data if d.get("content")
    ]
    print(f"📚 Tổng số điều luật: {len(docs)}")

    embedding_fn = Qwen3Embedding(model_embed, tokenizer_embed, device=DEVICE, batch_size=BATCH_SIZE)
    print("✨ Đang tạo mới Chroma DB ...")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    print(f"✅ Vector DB đã lưu thành công tại: {CHROMA_PATH}")

    test_emb = embedding_fn.embed_query("kiểm tra kích thước vector")
    print(f"✅ Embedding dimension: {len(test_emb)}")


if __name__ == "__main__":
    build_vector_db(force_rebuild=True)
