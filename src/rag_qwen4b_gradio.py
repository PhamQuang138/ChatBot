#!/usr/bin/env python3
import os
import re
import json
import torch
import numpy as np
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate


# ================== CONFIG ==================
BASE_DIR = "/home/quang/Documents/ChatBot"
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db_qwen_embed_vn")
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
THRESHOLD = 0.1
BATCH_SIZE = 4
# ===========================================


# ===== Embedding wrapper =====
class Qwen3Embedding(Embeddings):
    def __init__(self, model, tokenizer, device="cpu", batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def embed_documents(self, texts):
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                vecs = outputs.last_hidden_state[:, 0, :]
                vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(vecs.cpu().numpy())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size)).tolist()
        return np.vstack(all_embs).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ===== Utility =====
def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def embed_query_vector(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
    return emb.cpu().numpy()[0]


# ===== Initialization =====
vectordb = None
retriever = None
llm = None
prompt_template = None
embed_model = None
embed_tokenizer = None
embedding_fn = None

# ===== Initialization =====
def initialize_rag_components():
    global vectordb, retriever, llm, prompt_template, embed_model, embed_tokenizer, embedding_fn

    print("🛠️ Initializing RAG components...")

    # 1️⃣ Load embedding model
    print(f"🔹 Loading embedding model: {EMBED_MODEL}")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL, device_map="auto", torch_dtype=torch.float16)
    # ensure model is on the expected device for later inference
    try:
        embed_model.to(DEVICE)
    except Exception:
        # fallback: some HF models with device_map="auto" may not accept .to() - ignore if fails
        pass
    embedding_fn = Qwen3Embedding(embed_model, embed_tokenizer, DEVICE, BATCH_SIZE)
    print("✅ Embedding model ready.")

    # 2️⃣ Load Chroma DB
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"❌ Chroma DB not found at {CHROMA_PATH}")
    print(f"🔹 Loading Chroma DB from {CHROMA_PATH}")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    print("✅ Chroma retriever ready.")

    # 3️⃣ Load LLM
    print(f"🔹 Loading LLM {LLM_MODEL} (8-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config
    )
    llm_pipe = pipeline("text-generation", model=model_llm, tokenizer=tokenizer_llm,
                        max_new_tokens=512, do_sample=False, return_full_text=False)
    llm = HuggingFacePipeline(pipeline=llm_pipe)
    print("✅ LLM ready.")

    # 4️⃣ Prompt Template (phiên bản cực nghiêm ngặt)
    prompt_template = ChatPromptTemplate.from_template(
        """Bạn là **trợ lý pháp lý chuyên về Luật Dược Việt Nam**.

QUY TẮC NGHIÊM NGẶT:
- Chỉ trả lời dựa trên **nội dung điều luật trong CONTEXT** bên dưới.
- Nếu **không có thông tin phù hợp**, phải trả lời đúng câu này:
  👉 "Không tìm thấy thông tin này trong các điều luật được cung cấp."
- Không được thêm bất kỳ câu xin lỗi, suy luận hay lời giải thích nào khác.

---
### CONTEXT (Các điều luật liên quan)
{context}

### CÂU HỎI
{question}

### TRẢ LỜI (ngắn gọn, chuẩn pháp lý, tiếng Việt)
"""
    )

    print("✅ All components initialized.\n")


def rag_query(question: str):
    if not vectordb or not llm:
        return "⚠️ RAG chưa được khởi tạo đúng cách.", ""

    # === Nếu có dạng "Điều X" ===
    match = re.search(r"Điều\s*(\d+)", question, re.IGNORECASE)
    if match:
        article_num = match.group(1).strip()
        all_data = vectordb._collection.get(include=["documents", "metadatas"], limit=10000)

        found_docs = []
        for doc, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
            # --- Lấy thông tin article ---
            art = ""
            if isinstance(meta, dict):
                art = meta.get("article", "") or meta.get("source", "")
            if not art:
                continue

            # --- So khớp theo số điều ---
            m = re.search(r"(\d+)", str(art))
            if m and m.group(1).strip() == article_num:
                # ✅ Nội dung nằm trong `documents`, không phải `meta["content"]`
                content_text = doc
                if content_text:
                    found_docs.append(f"{art}\n{content_text.strip()}")

        # --- Không tìm thấy ---
        if not found_docs:
            return "Không tìm thấy thông tin này trong các điều luật được cung cấp.", f"Điều {article_num} (không thấy trong DB)"

        # --- Ghép context và gọi LLM ---
        context = "\n---\n".join(found_docs)
        prompt = prompt_template.format(context=context, question=question)
        print("\n===== DEBUG PROMPT =====\n", prompt[:1500], "\n=========================\n")

        answer = llm.invoke(prompt).strip()

        if not answer or len(answer) < 5:
            return "Không tìm thấy thông tin này trong các điều luật được cung cấp.", f"Điều {article_num} (có {len(found_docs)} đoạn)"
        return answer, f"Điều {article_num} (tìm thấy {len(found_docs)} đoạn)"

    # === Nếu là câu hỏi tự nhiên ===
    docs = retriever.invoke(question)
    if not docs:
        return "⚠️ Không tìm thấy điều luật liên quan.", ""

    q_vec = embed_query_vector(question, embed_tokenizer, embed_model)
    ranked_docs = []
    for d in docs:
        d_inputs = embed_tokenizer(d.page_content, return_tensors="pt", truncation=True,
                                   padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = embed_model(**d_inputs)
            vec = outputs.last_hidden_state[:, 0, :]
            vec = vec / vec.norm(dim=-1, keepdim=True)
        d_vec = vec.cpu().numpy()[0]
        score = cosine_similarity(q_vec, d_vec)
        if score >= THRESHOLD:
            ranked_docs.append((score, d))

    ranked_docs.sort(key=lambda x: x[0], reverse=True)
    if not ranked_docs:
        return "⚠️ Không có điều luật nào vượt ngưỡng tương đồng.", ""

    context_blocks, sources = [], []
    for score, d in ranked_docs[:3]:
        art = d.metadata.get("article", "Không rõ")
        context_blocks.append(f"[{score:.2f}] {art}\n{d.page_content}")
        sources.append(f"{art} (độ tương đồng={score:.2f})")

    context = "\n---\n".join(context_blocks)
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt).strip()

    if not answer or "không tìm thấy" in answer.lower() or len(answer) < 5:
        answer = "Không tìm thấy thông tin này trong các điều luật được cung cấp."

    return answer, "\n".join(sources)



# ======= Startup =======
try:
    initialize_rag_components()
except Exception as e:
    print(f"❌ LỖI KHỞI TẠO NGHIÊM TRỌNG: {e}")


# ======= Gradio UI =======
with gr.Blocks(title="⚖️ Trợ lý pháp lý Luật Dược Việt Nam (Qwen RAG)") as demo:
    gr.Markdown(f"""
    ## ⚖️ Trợ lý pháp lý Luật Dược Việt Nam
    **LLM:** `{LLM_MODEL}`  
    **Embedding:** `{EMBED_MODEL}`  
    **Thiết bị:** `{DEVICE}`  
    **CSDL:** `{CHROMA_PATH}`
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Nhập câu hỏi pháp lý:", lines=3,
                                  placeholder="Ví dụ: Điều 47 quy định gì về thuốc generic?")
            ask = gr.Button("Hỏi", variant="primary")
            clear = gr.Button("Xoá")
        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Trả lời", lines=10, interactive=False)
            source_box = gr.Textbox(label="Điều luật trích dẫn", lines=6, interactive=False)

    ask.click(fn=rag_query, inputs=question, outputs=[answer_box, source_box])
    clear.click(lambda: ("", "", ""), outputs=[question, answer_box, source_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
