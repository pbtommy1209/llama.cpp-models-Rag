# rag_easy_preprocess.py
# Minimal preprocessing: read FAQ jsonl -> make chunks -> build FAISS + BM25
# Usage:
#   python rag_easy_preprocess.py --in data/faq_raw.jsonl --outdir artifacts_easy
# If faq_raw.jsonl doesn't exist, it will fallback to faq_docs.jsonl

import os, json, argparse, re, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def tokenize_simple(s: str):
    # simple whitespace tokenizer (works fine for English)
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

def build_dense(texts, model_name):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    return index, embs

def build_bm25(texts):
    tok = [tokenize_simple(t) for t in texts]
    return BM25Okapi(tok), tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/faq_raw.jsonl")
    ap.add_argument("--fallback", default="faq_docs.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--outdir", default="artifacts_easy")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[info] {inp} not found, falling back to {args.fallback}")
        inp = Path(args.fallback)
    rows = read_jsonl(str(inp))
    if not rows:
        raise SystemExit("No data found. Make sure faq_raw.jsonl or faq_docs.jsonl exists.")

    # Make simple chunks: "Q: ... A: ..."
    chunks = []
    for r in rows:
        if "text" in r:
            chunks.append(normalize_text(r["text"]))
        else:
            q = r.get("question","").strip()
            a = r.get("answer","").strip()
            if q or a:
                chunks.append(normalize_text(f"Q: {q}\nA: {a}"))

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "chunks.jsonl"), "w", encoding="utf-8") as w:
        for i, t in enumerate(chunks):
            w.write(json.dumps({"chunk_id": i, "text": t}, ensure_ascii=False) + "\n")

    # Dense
    index, embs = build_dense(chunks, args.model)
    faiss.write_index(index, os.path.join(args.outdir, "dense.index"))
    np.save(os.path.join(args.outdir, "dense.npy"), embs)

    # BM25
    bm25, tok = build_bm25(chunks)
    with open(os.path.join(args.outdir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "tok": tok}, f)

    # meta
    with open(os.path.join(args.outdir, "meta.pkl"), "wb") as f:
        pickle.dump([{"chunk_id": i, "text": t} for i, t in enumerate(chunks)], f)

    print(f"✅ Done. {len(chunks)} chunks")
    print(f"Artifacts in: {args.outdir}")

if __name__ == "__main__":
    main()
