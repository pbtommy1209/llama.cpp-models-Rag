# rag_easy_ask.py
# Retrieve top chunks (hybrid dense + BM25) and generate answer with Ollama
# Usage:
#   OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=llama3.2:latest python rag_easy_ask.py --q "How to reset my password?"

import os, json, argparse
import numpy as np
import requests
import faiss, pickle

ART = "artifacts_easy"
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

def load_dense():
    idx = faiss.read_index(os.path.join(ART, "dense.index"))
    embs = np.load(os.path.join(ART, "dense.npy"))
    return idx, embs.shape[1]

def load_bm25():
    with open(os.path.join(ART, "bm25.pkl"), "rb") as f:
        return pickle.load(f)["bm25"]

def load_meta():
    with open(os.path.join(ART, "meta.pkl"), "rb") as f:
        return pickle.load(f)

def embed_query(q, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return m.encode([q], normalize_embeddings=True).astype(np.float32)

def dense_search(index, qv, k=10):
    D,I = index.search(qv, k)
    return I[0].tolist(), D[0].tolist()

def bm25_search(bm25, q, k=10):
    import re
    toks = re.findall(r"[A-Za-z0-9_]+", q.lower())
    scores = bm25.get_scores(toks)
    idx = np.argsort(scores)[::-1][:k]
    return idx.tolist(), [scores[i] for i in idx]

def hybrid(di, ds, bi, bs, alpha=0.5, k=10):
    from collections import defaultdict
    S = defaultdict(float)
    if ds:
        dmax = max(ds) or 1e-6
        for i,s in zip(di, ds): S[i] += alpha*(s/dmax)
    if bs:
        bmax = max(bs) or 1e-6
        for i,s in zip(bi, bs): S[i] += (1-alpha)*(s/bmax)
    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:k]
    return [i for i,_ in ranked]

def build_prompt(q, ctx):
    joined = "\n\n".join([f"[{i}] {t}" for i,t in enumerate(ctx)])
    system = "You are a helpful assistant. Use ONLY the context to answer. If not found, say you don't know."
    user = f"Context:\n{joined}\n\nQuestion: {q}\nAnswer:"
    return system, user

def ollama_chat(model, system, user, temperature=0.2, max_tokens=256):
    url = f"{OLLAMA}/api/chat"
    body = {
        "model": model.strip(),
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "options": {"temperature": temperature},
        "stream": False
    }
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Your question")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--topk_ctx", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    idx, dim = load_dense()
    bm25 = load_bm25()
    meta = load_meta()

    qv = embed_query(args.q, args.embed_model)
    di, ds = dense_search(idx, qv, args.topk)
    bi, bs = bm25_search(bm25, args.q, args.topk)
    I = hybrid(di, ds, bi, bs, args.alpha, k=args.topk)
    ctx = [meta[i]["text"] for i in I[:args.topk_ctx]]

    sys, usr = build_prompt(args.q, ctx)
    ans = ollama_chat(MODEL, sys, usr)
    print("=== Answer ===")
    print(ans)
    print("\n=== Context used ===")
    for i, t in enumerate(ctx): print(f"[{i}] {t[:180]}{'...' if len(t)>180 else ''}")

if __name__ == "__main__":
    main()
