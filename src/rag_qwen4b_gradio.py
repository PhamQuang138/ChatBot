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
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"        # ✅ Đã thay 3B
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 20
THRESHOLD = 0.005
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

    # 3️⃣ Load LLM
    print(f"🔹 Loading LLM {LLM_MODEL} (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )

    # ⚡ Load LoRA adapter (nếu có)
    lora_path = os.path.join(BASE_DIR, "src", "lora_qwen_druglaw_4bit")
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
        return_full_text=False
    )

    print("✅ LLM ready.")

prompt_template_normal = ChatPromptTemplate.from_template(
        """Bạn là trợ lý pháp lý chuyên về **Luật Dược Việt Nam**.

    Dựa **chỉ trên phần CONTEXT dưới đây**, hãy **trích nguyên văn quy định pháp luật** có liên quan để trả lời câu hỏi.
    Không được:
    - thêm bình luận, suy luận, hay diễn giải.
    - liệt kê các lựa chọn kiểu a), b), c) nếu câu hỏi không yêu cầu.
    - tự đánh giá hay chọn đáp án.

    ---
    📘 CONTEXT:
    {context}

    💬 CÂU HỎI:
    {question}

    ✍️ TRẢ LỜI (trích nguyên văn quy định):
    """
    )

prompt_template_quiz = ChatPromptTemplate.from_template(
        """Bạn là trợ lý pháp lý chuyên về **Luật Dược Việt Nam**.

    Câu hỏi sau đây có dạng **trắc nghiệm nhiều lựa chọn** (a, b, c, d...).
    "Chỉ trả lời các mục a) tới h) đã cho, KHÔNG sinh thêm nhãn hay A:, B:, C: trống."
    Dựa **chỉ trên phần CONTEXT**, hãy:
    - trích nguyên văn quy định liên quan, 
    - sau đó **chỉ ra đáp án đúng duy nhất**, không thêm giải thích hay bình luận.

    ---
    📘 CONTEXT:
    {context}

    💬 CÂU HỎI (trắc nghiệm):
    {question}

    ✍️ TRẢ LỜI (nguyên văn + chọn đáp án đúng):
    """
    )

print("✅ All components initialized.\n")

def rag_query(question: str, use_llm: bool = True):
    if not vectordb or not llm:
        return "⚠️ RAG chưa được khởi tạo đúng cách.", ""

    # --- 1️⃣ Nếu người dùng hỏi theo "Điều X"
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
        if not use_llm:
            return context, f"Điều {article_num} (tìm thấy {len(found_docs)} đoạn)"

        # Giữ câu hỏi tự nhiên, không ép prompt nữa
        question = f"Nội dung quy định tại Điều {article_num} là gì?"

    # --- 2️⃣ Hybrid Search (BM25 + Semantic)
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

    # --- 3️⃣ Chọn điều có điểm cao nhất
    merged.sort(key=lambda x: x[0], reverse=True)
    best_score, _, best_art = merged[0]
    same_articles = [doc for score, doc, art in merged if art == best_art]

    cleaned_content = "\n".join(dict.fromkeys("\n".join(same_articles).splitlines()))
    context = f"{best_art}\n{cleaned_content.strip()}"
    if len(context.split()) > 4000:
        context = " ".join(context.split()[:4000])

    if not use_llm:
        return context, f"{best_art} (score={best_score:.2f})"

    # --- 4️⃣ Tạo prompt phù hợp ---
    if re.search(r"\b[a-e]\)", question.lower()):
        prompt_text = prompt_template_quiz.format(context=context, question=question)
    else:
        prompt_text = prompt_template_normal.format(context=context, question=question)

    try:
        result = llm(prompt_text,max_new_tokens=512,do_sample=True,temperature=0.1,top_p=0.8)
        answer = result[0]["generated_text"].strip()
        answer = re.sub(r'(?i)assistant[:：-]*\s*', '', answer).strip()
        # ❌ Cắt phần "Explanation" hoặc "Giải thích" nếu có
        answer = re.split(r"(###?\s*Explanation:|Giải thích[:：])", answer, flags=re.IGNORECASE)[0].strip()

        # ❌ Cắt phần "Answer:" nếu có tiêu đề
        answer = re.sub(r"^###?\s*Answer:\s*", "", answer, flags=re.IGNORECASE).strip()

        # ❌ Loại bỏ tiêu đề "Trả lời" hoặc phần lặp lại
        answer = re.sub(r"(?i)(###?\s*trả lời[:：]*\s*)", "", answer).strip()

        # ✅ Cắt bỏ phần trùng lặp nếu mô hình lặp nội dung nhiều lần
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)

        # ✅ Giữ lại tối đa 1 đoạn nội dung trùng lặp (tránh 5–6 lần lặp y hệt)
        answer = "\n".join(unique_lines)

        # ✅ Nếu mô hình tự sinh nhiều khối “---”, cắt phần đầu tiên
        answer = answer.split('---')[0].strip()

        # ✅ Nếu mô hình lặp lại toàn bộ block nhiều lần, cắt phần lặp dựa trên dòng đầu tiên
        if answer.count(unique_lines[0]) > 1:
            first = answer.find(unique_lines[0])
            second = answer.find(unique_lines[0], first + len(unique_lines[0]))
            if second != -1:
                answer = answer[:second].strip()

    except Exception as e:
        answer = f"Lỗi khi sinh câu trả lời: {e}"

    # Nếu LLM không trích được — trả context để debug
    if not answer or "không tìm thấy" in answer.lower():
        return context, f"[DEBUG: LLM không trích được] {best_art} (score={best_score:.2f})"

    return answer, f"{best_art} (score={best_score:.2f})"

try:
    initialize_rag_components()
except Exception as e:
    print(f"❌ LỖI KHỞI TẠO NGHIÊM TRỌNG: {e}")


# ======= Gradio UI =======
with gr.Blocks(title="⚖️ Trợ lý pháp lý Luật Dược Việt Nam (Qwen 3B RAG)") as demo:
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
