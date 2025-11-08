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
from rank_bm25 import BM25Okapi

# ================== CONFIG ==================
BASE_DIR = "/home/quang/Documents/ChatBot"
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db_qwen_embed_vn")
LLM_MODEL = "meta-llama/Llama-3.2-1B"      # ✅ model base đúng của LoRA fine-tuned
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # vẫn có thể dùng Qwen3 embed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 20
THRESHOLD = 0.005
BATCH_SIZE = 4

import difflib

def remove_near_duplicates(lines, similarity=0.9):
    cleaned = []
    for line in lines:
        if not cleaned:
            cleaned.append(line)
            continue
        sim = difflib.SequenceMatcher(None, cleaned[-1], line).ratio()
        if sim < similarity:
            cleaned.append(line)
    return cleaned

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

def initialize_rag_components():
    global vectordb, retriever, llm, prompt_template, embed_model, embed_tokenizer, embedding_fn

    print("🛠️ Initializing RAG components...")

    # 1️⃣ Load embedding model
    print(f"🔹 Loading embedding model: {EMBED_MODEL}")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(
        EMBED_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    try:
        embed_model.to(DEVICE)
    except Exception:
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

    # 3️⃣ Load LLM (base Llama + LoRA)
    print(f"🔹 Loading base LLM: {LLM_MODEL} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )

    # ⚡ Load LoRA adapter
    lora_path = os.path.join(BASE_DIR, "src", "lora_llama3_4bit")
    if os.path.exists(lora_path):
        try:
            from peft import PeftModel
            print(f"🔹 Attaching LoRA adapter from {lora_path}")
            model_llm = PeftModel.from_pretrained(model_llm, lora_path)
            print("✅ LoRA adapter loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Không thể load LoRA adapter: {e}")
    else:
        print(f"⚠️ Không tìm thấy thư mục LoRA tại {lora_path}")

    llm = pipeline(
        "text-generation",
        model=model_llm,
        tokenizer=tokenizer_llm,
        return_full_text=False,
        no_repeat_ngram_size=6
    )

    print("✅ LLM ready.")


prompt_template_normal = ChatPromptTemplate.from_template(
    """Bạn là trợ lý pháp lý chuyên về **Luật Dược Việt Nam**.

Dựa **chỉ trên phần CONTEXT dưới đây**, hãy **trích nguyên văn quy định pháp luật** có liên quan để trả lời câu hỏi.
- Nếu trong phần CONTEXT có các câu đánh số (1, 2, 3...) hoặc a),b),c),...hãy trình bày xuống dòng rõ ràng.

Tuyệt đối **không được suy luận, diễn giải, hoặc paraphrase**.

- Nếu không có nội dung nào trong CONTEXT trùng khớp hoặc trích dẫn nguyên văn điều luật, **dù có các câu tương tự hoặc diễn giải**, thì phải trả đúng duy nhất:
  "Không tìm thấy quy định liên quan trong CONTEXT."

- Nếu có nhiều đoạn giống nhau hoặc trùng lặp, chỉ giữ lại **một bản đầy đủ nhất**.

---
📑 CONTEXT:
{context}

💬 CÂU HỎI:
{question}

✍️ TRẢ LỜI (trích nguyên văn quy định hoặc câu thông báo trên):
"""
)


prompt_template_quiz = ChatPromptTemplate.from_template(
    """Bạn là trợ lý pháp lý chuyên về **Luật Dược Việt Nam**.

Câu hỏi sau đây có dạng **trắc nghiệm nhiều lựa chọn** (a, b, c, d...).
Dựa **chỉ trên phần CONTEXT**, hãy:
- trích nguyên văn quy định liên quan,
- không thêm giải thích hay bình luận.
- Nếu trong phần CONTEXT có các câu đánh số (1, 2, 3...) hoặc a),b),c),...hãy trình bày xuống dòng rõ ràng.


Không được:
- Tự tạo nội dung, URL, hay số liệu.
- Dịch sang ngôn ngữ khác.
- Thêm bình luận, giải thích hay suy luận.

---
📘 CONTEXT:
{context}

💬 CÂU HỎI (trắc nghiệm):
{question}

✍️ TRẢ LỜI (nguyên văn + chọn đáp án đúng):
"""
)

print("✅ All components initialized.\n")

# ============= RAG QUERY =============
def rag_query(question: str, use_llm: bool = True):
    if not vectordb or not llm:
        return "⚠️ RAG chưa được khởi tạo đúng cách.", ""

    # === 1️⃣ Nếu câu hỏi có chứa 'Điều X' → chỉ truy xuất dữ liệu, không gọi LLM ===
    match = re.search(r"Điều\s*(\d+)", question.strip(), re.IGNORECASE)
    if match:
        article_num = match.group(1).strip()
        all_data = vectordb._collection.get(include=["documents", "metadatas"], limit=10000)

        found_docs = []
        for doc, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
            art = meta.get("article", "") if isinstance(meta, dict) else ""
            m = re.search(r"(\d+)", str(art))
            if m and m.group(1).strip() == article_num:
                found_docs.append(f"{art}\n{doc.strip()}")

        if not found_docs:
            return "Không tìm thấy thông tin này trong các điều luật.", f"Điều {article_num} (không thấy trong DB)"

        context = "\n---\n".join(found_docs)

        # 🚫 Tự động bỏ qua LLM nếu câu hỏi chỉ dạng 'Điều X' hoặc tương tự
        if re.fullmatch(r".*Điều\s*\d+.*", question.strip(), re.IGNORECASE):
            return context, f"Điều {article_num} (tìm thấy {len(found_docs)} đoạn)"

        # Nếu câu hỏi dài hoặc có thêm nội dung → vẫn có thể gọi LLM
        if not use_llm:
            return context, f"Điều {article_num} (tìm thấy {len(found_docs)} đoạn)"

        question = f"Nội dung quy định tại Điều {article_num} là gì?"

    # === 2️⃣ Hybrid Search (BM25 + Embedding) ===
    all_data = vectordb._collection.get(include=["documents", "metadatas"], limit=10000)
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    if not documents:
        return "⚠️ CSDL trống hoặc chưa tải đúng.", ""

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(question.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:TOP_K]
    bm25_docs = [(bm25_scores[i], documents[i], metadatas[i]) for i in top_bm25_idx if bm25_scores[i] > 0]

    sem_docs = retriever.invoke(question)
    merged, seen = [], set()

    for score, doc, meta in bm25_docs:
        art = meta.get("article", "Không rõ") if isinstance(meta, dict) else "Không rõ"
        if doc not in seen:
            merged.append((float(score), doc, art))
            seen.add(doc)

    q_vec = embed_query_vector(question, embed_tokenizer, embed_model)
    for d in sem_docs:
        d_vec = embed_query_vector(d.page_content, embed_tokenizer, embed_model)
        cos_sim = cosine_similarity(q_vec, d_vec)
        art = d.metadata.get("article", "Không rõ")
        if d.page_content not in seen and cos_sim >= THRESHOLD:
            merged.append((float(cos_sim), d.page_content, art))
            seen.add(d.page_content)

    if not merged:
        return "⚠️ Không tìm thấy điều luật liên quan.", ""

    merged.sort(key=lambda x: x[0], reverse=True)
    best_score, _, best_art = merged[0]
    same_articles = [doc for score, doc, art in merged if art == best_art]
    context = f"{best_art}\n" + "\n".join(same_articles).strip()

    if len(context.split()) > 4000:
        context = " ".join(context.split()[:4000])

    # Nếu không muốn gọi LLM thì trả lại luôn context
    if not use_llm:
        return context, f"{best_art} (score={best_score:.2f})"

    # === 3️⃣ Gọi LLM nếu cần ===
    prompt_text = (
        prompt_template_quiz.format(context=context, question=question)
        if re.search(r"\b[a-e]\)", question.lower())
        else prompt_template_normal.format(context=context, question=question)
    )

    try:
        result = llm(prompt_text, max_new_tokens=512)
        answer = result[0]["generated_text"].strip()

        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        unique_lines = remove_near_duplicates(lines, similarity=0.9)
        answer = " ".join(unique_lines)

    except Exception as e:
        answer = f"Lỗi khi sinh câu trả lời: {e}"

    return answer, f"{best_art} (score={best_score:.2f})"


# ======= Gradio UI =======
try:
    initialize_rag_components()
except Exception as e:
    print(f"❌ LỖI KHỞI TẠO NGHIÊM TRỌNG: {e}")

with gr.Blocks(title="⚖️ Trợ lý pháp lý Luật Dược Việt Nam (Llama 1B LoRA)") as demo:
    gr.Markdown(f"""
    ## ⚖️ Trợ lý pháp lý Luật Dược Việt Nam
    **LLM:** `{LLM_MODEL}`  
    **Embedding:** `{EMBED_MODEL}`  
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Nhập câu hỏi pháp lý:", lines=3,
                                  placeholder="Ví dụ: Điều 47 quy định gì về thuốc generic?")
            use_llm = gr.Checkbox(label="Gọi LLM (bật để sinh câu trả lời)", value=True)
            ask = gr.Button("Hỏi", variant="primary")
            clear = gr.Button("Xoá")
        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Trả lời", lines=10, interactive=False)
            source_box = gr.Textbox(label="Điều luật trích dẫn", lines=6, interactive=False)

    ask.click(fn=rag_query, inputs=[question, use_llm], outputs=[answer_box, source_box])
    clear.click(lambda: ("", "", ""), outputs=[question, answer_box, source_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
