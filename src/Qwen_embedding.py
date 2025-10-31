class Qwen_embedding:
    def __init__(self, model, tokenizer, device="cpu", batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.last_hidden_state[:, 0, :]  # CLS pooling
                batch_emb = last_hidden / last_hidden.norm(dim=-1, keepdim=True)
                embeddings.extend(batch_emb.cpu().numpy())
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]
