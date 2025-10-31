#!/usr/bin/env python3
import os
import torch
import gradio as gr
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    AutoModel
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
import re
# ================== CONFIG ==================
BASE_DIR = "/home/quang/Documents/ChatBot"
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db_qwen_embed")
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 15
THRESHOLD = 0.05
BATCH_SIZE = 4
# ===========================================

# ===== Embedding wrapper (LangChain-style) =====
class Qwen3Embedding(Embeddings):
    """Thin wrapper so Chroma/langchain can call embed_documents/embed_query.
    Internally uses a HF AutoModel + AutoTokenizer passed in at init.
    """
    def __init__(self, model, tokenizer, device="cpu", batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def embed_documents(self, texts):
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                vecs = outputs.last_hidden_state[:, 0, :]
                vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(vecs.cpu().numpy())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size)).tolist()
        return np.vstack(all_embs).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ===== Globals =====
vectordb = None
llm = None
prompt_template = None
retriever = None
embed_tokenizer = None
embed_model = None
embedding_fn = None


# ===== Initialization =====
def initialize_rag_components():
    global vectordb, llm, prompt_template, retriever, embed_tokenizer, embed_model, embedding_fn

    print("🛠️ Initializing RAG components...")

    # 1️⃣ Load embedding model (for query if needed) and wrap for Chroma
    print(f"🔹 Loading embedding model for query: {EMBED_MODEL}")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(
        EMBED_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    embedding_fn = Qwen3Embedding(model=embed_model, tokenizer=embed_tokenizer, device=DEVICE, batch_size=BATCH_SIZE)
    print("✅ Embedding model ready for semantic scoring.")

    # 2️⃣ Load Chroma DB (use prebuilt embeddings) and retriever
    print(f"🔹 Loading Chroma DB from {CHROMA_PATH}")
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"❌ Chroma DB not found at {CHROMA_PATH}. Please run build_chromaDB.py first.")
    # Provide embedding_function so Chroma uses same dimensionality & won't fallback to onnx/minilm
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    print("✅ Chroma retriever ready.")

    # 2.5 Verify dimension safely (try to include embeddings)
    try:
        sample_doc = vectordb._collection.get(limit=1, include=["embeddings"])
        if not sample_doc or not sample_doc.get("embeddings"):
            # Could be fine (Chroma may not return embeddings by default); log and continue.
            print("⚠️ Couldn't fetch embeddings from collection.get(...). Continuing without explicit dimension check.")
        else:
            db_dim = len(sample_doc["embeddings"][0])
            test_emb = embedding_fn.embed_query("dimension test")
            model_dim = len(test_emb)
            if db_dim != model_dim:
                raise ValueError(f"❌ Dimension mismatch: DB={db_dim}, Model={model_dim}. Rebuild DB with same model.")
            print(f"✅ Verified dimension match: {db_dim}D.")
    except Exception as e:
        # Non-fatal; we allow launch even if strict check fails; report error for debugging.
        print(f"⚠️ Dimension check skipped or failed (non-fatal): {e}")

    # 3️⃣ Load LLM (8-bit quantized) for generation
    print(f"🔹 Loading LLM {LLM_MODEL} (8-bit quantized)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
    llm_pipe = pipeline(
        "text-generation",
        model=model_llm,
        tokenizer=tokenizer_llm,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
        eos_token_id=tokenizer_llm.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipe)
    print("✅ LLM ready.")

    # 4️⃣ Prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """You are a legal assistant specialized in Vietnamese pharmacy law.
    Answer the following question **based only on the provided legal articles**.
    If the information is **not found** in the articles, clearly say:
    "I cannot find this information in the provided context".Don’t answer anything outside Vietnamese Pharmacy Law context.
    
    ---
    ### CONTEXT (Extracted Legal Articles)
    {context}
    
    ### QUESTION
    {question}
    
    ### INSTRUCTIONS
    - Summarize concisely and accurately.
    - Quote or refer to article numbers when possible (e.g., “According to Article 47…”).
    - Do NOT invent information not present in the context.
    - Use clear, professional English in your answer.
    
    ### ANSWER
    """
    )
    print("✅ All components initialized.\n")


# ===== Utilities =====
def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_query_vector(text: str):
    """Embed a single query using embed_model/tokenizer (returns numpy 1D vector)."""
    global embed_tokenizer, embed_model
    if embed_tokenizer is None or embed_model is None:
        raise RuntimeError("Embedding model/tokenizer not initialized.")
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
    return emb.cpu().numpy()[0]



def rag_query(question: str):
    if not retriever or not llm:
        return "⚠️ RAG not initialized properly.", ""

    # 1️⃣ Kiểm tra nếu user hỏi theo dạng "Article XXX" hoặc "Điều XXX"
    match = re.search(r"(?:Article|Điều)\s*(\d+)", question, re.IGNORECASE)
    if match:
        article_num = match.group(1)
        # Truy xuất trực tiếp từ Chroma bằng metadata
        all_data = vectordb._collection.get(include=["documents", "metadatas"], limit=10000)
        results = {"documents": [], "metadatas": []}

        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            article_field = meta.get("article", "")
            if f"Article {article_num}" in article_field:
                results["documents"].append(doc)
                results["metadatas"].append(meta)

        if results and len(results.get("documents", [])) > 0:
            content = results["documents"][0]
            meta = results["metadatas"][0] if results.get("metadatas") else {}
            art = meta.get("article", f"Article {article_num}")
            context = f"{art}\n{content}"

            prompt = prompt_template.format(context=context, question=question)
            answer = llm.invoke(prompt)
            if isinstance(answer, str):
                answer = answer.strip()

                # 🚫 Xóa phần "Note:" hoặc chú thích tương tự
                answer = re.sub(r"(?i)note\s*:.*", "", answer)
                answer = re.sub(r"\n{2,}", "\n\n", answer).strip()

            return answer, f"{art} (direct lookup)"
        else:
            return f"⚠️ Article {article_num} not found in database.", ""

    # 2️⃣ Nếu không phải dạng “Article N” → chạy RAG bình thường
    docs = retriever.invoke(question)
    if not docs:
        return "⚠️ No relevant documents found.", ""

    q_vec = embed_query_vector(question)
    ranked_docs = []
    for d in docs:
        d_vec = embed_tokenizer(
            d.page_content, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            outputs = embed_model(**d_vec)
            vec = outputs.last_hidden_state[:, 0, :]
            vec = vec / vec.norm(dim=-1, keepdim=True)
        d_vec = vec.cpu().numpy()[0]

        score = cosine_similarity(q_vec, d_vec)
        if score >= THRESHOLD:
            ranked_docs.append((score, d))

    ranked_docs.sort(key=lambda x: x[0], reverse=True)
    if not ranked_docs:
        return "⚠️ No relevant context passed the similarity threshold.", ""

    context_blocks, sources = [], []
    for score, d in ranked_docs[:3]:
        art = d.metadata.get("article", "N/A")
        context_blocks.append(f"[{score:.2f}] {art}\n{d.page_content}")
        sources.append(f"{art} (score={score:.2f})")

    context = "\n---\n".join(context_blocks)
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)

    if isinstance(answer, str):
        answer = answer.strip()

        # 🔧 Loại bỏ lặp lại nếu model trả về đoạn giống nhau nhiều lần
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        deduped = []
        for line in lines:
            if not deduped or line != deduped[-1]:
                deduped.append(line)
        answer = "\n".join(deduped)

        # 🚫 Xóa phần "Note:" hoặc các chú thích tương tự
        answer = re.sub(r"(?i)note\s*:.*", "", answer)
        answer = re.sub(r"\n{2,}", "\n\n", answer).strip()

    return answer, "\n".join(sources)


# ======= Startup =======
try:
    initialize_rag_components()
except Exception as e:
    print(f"❌ FATAL INIT ERROR: {e}")

# ======= Gradio UI =======
with gr.Blocks(title="⚖️ Qwen RAG Legal Chat") as demo:
    gr.Markdown(
        f"""
        ## ⚖️ Qwen RAG Legal Information System (Smart RAG)
        **LLM:** `{LLM_MODEL}`  
        **Embedding:** `{EMBED_MODEL}`  
        **Device:** `{DEVICE}`  
        **DB Path:** `{CHROMA_PATH}`  
        ---
        """
    )

    if not retriever:
        gr.Warning("⚠️ Initialization failed — check logs and DB path.")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Enter a legal/pharmacy question:",
                lines=3,
                placeholder="e.g. What are the conditions for opening a pharmacy?"
            )
            ask = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")
        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Answer", lines=10, interactive=False)
            source_box = gr.Textbox(label="Sources & Relevance", lines=6, interactive=False)

    ask.click(fn=rag_query, inputs=question, outputs=[answer_box, source_box])
    clear.click(lambda: ("", "", ""), outputs=[question, answer_box, source_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
