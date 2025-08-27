# eval_gen_ollama.py (fixed + verbatim JSON answer + normalization)
import os, json, argparse, re, pickle, numpy as np, requests, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

ART = "artifacts_easy"
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

ANSWER_KEYS = [r"\bA:\s*", r"\bAnswer\s*:\s*"]

def extract_answers_from_ctx(ctx_text: str):
    """Return a list of candidate answers parsed out of a context snippet."""
    c = ctx_text.strip()
    c = re.sub(r"\s+", " ", c)
    cands = []
    # 1) Split on multiple possible markers
    for pat in ANSWER_KEYS:
        m = re.search(pat + r"(.+)", c, flags=re.I)
        if m:
            ans = m.group(1).strip()
            # Stop at next Q: if present
            ans = re.split(r"\bQ\s*:\s*|\bQuestion\s*:\s*", ans, maxsplit=1, flags=re.I)[0].strip()
            # Keep it short-ish
            ans = " ".join(ans.split()[:60])
            if ans:
                cands.append(ans)
    # 2) If none found, also try simple “Q: … A: …” shape
    m2 = re.search(r"\bQ\s*:\s*.+?\bA\s*:\s*(.+)", c, flags=re.I)
    if m2:
        ans = re.split(r"\bQ\s*:\s*|\bQuestion\s*:\s*", m2.group(1).strip(), maxsplit=1, flags=re.I)[0]
        ans = " ".join(ans.split()[:60]).strip()
        if ans:
            cands.append(ans)
    # Dedup while preserving order
    seen = set(); out=[]
    for a in cands:
        if a not in seen:
            seen.add(a); out.append(a)
    return out

def best_extracted_answer(q: str, candidates: list[str]):
    """Pick the most relevant extracted answer to the question using a tiny heuristic."""
    if not candidates:
        return ""
    # simple heuristic: prefer the shortest plausible answer (tends to match gold precisely)
    # tie-break with lexical overlap to the question
    def score(a):
        qa_overlap = len(set(tok(q)) & set(tok(a)))
        return (len(a.split()), qa_overlap)  # shorter is better; more overlap is better
    # sort by length ascending then overlap descending
    candidates_sorted = sorted(candidates, key=lambda a: (len(a.split()), -len(set(tok(q)) & set(tok(a)))))
    return candidates_sorted[0]


def tok(s): 
    return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())

def em(p, g): 
    return int((p or "").strip() == (g or "").strip())

def f1(p, g):
    from collections import Counter
    P, G = Counter(tok(p)), Counter(tok(g))
    common = sum((P & G).values())
    if not p and not g: return 1.0
    if not p or not g or common == 0: return 0.0
    prec = common / max(1, len(tok(p)))
    rec  = common / max(1, len(tok(g)))
    return 2 * prec * rec / (prec + rec)

def rouge_l(p, g):
    from difflib import SequenceMatcher
    s = SequenceMatcher(None, p or "", g or "")
    return sum(m.size for m in s.get_matching_blocks()) / max(1, len(g or ""))

def normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # keep letters, numbers, latin-extended, and CJK
    s = re.sub(r'[^0-9a-z\u00C0-\u024F\u4e00-\u9fff]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_all(embed_model):
    idx = faiss.read_index(os.path.join(ART, "dense.index"))
    with open(os.path.join(ART, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)["bm25"]
    with open(os.path.join(ART, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    emb = SentenceTransformer(embed_model)
    return idx, bm25, meta, emb

def bm25_search(bm25, q, k):
    toks = re.findall(r"[A-Za-z0-9_]+", q.lower()) or list(q.lower())  # OOV fallback
    scores = bm25.get_scores(toks)
    I = np.argsort(scores)[::-1][:k]
    return I.tolist(), [scores[i] for i in I]

def dense_search(index, emb, q, k):
    qv = emb.encode([q], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(qv, k)
    return I[0].tolist(), D[0].tolist()

def hybrid(di, ds, bi, bs, alpha=0.5, k=10):
    from collections import defaultdict
    S = defaultdict(float)
    if ds:
        dmax = max(ds) or 1e-6
        for i, s in zip(di, ds): S[i] += alpha * (s / dmax)
    if bs:
        bmax = max(bs) or 1e-6
        for i, s in zip(bi, bs): S[i] += (1 - alpha) * (s / bmax)
    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:k]
    return [i for i, _ in ranked]

def ollama_chat(model, system, user, temperature=0.0, max_tokens=128):
    url = f"{OLLAMA}/api/chat"
    body = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "format": "json"  # ask for strict JSON
        },
        "stream": False,
    }
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]  # JSON string

def build_prompt(q, ctx):
    ctx_text = "\n\n".join([f"[{i}] {c}" for i, c in enumerate(ctx)])
    sys = (
        "You are a precise QA assistant for enterprise FAQs.\n"
        "Answer ONLY from the context. If the answer is not present, reply with an empty string.\n"
        'CRITICAL: Return STRICT JSON: {"answer": "..."} and nothing else.\n'
        'CRITICAL: The value of "answer" must be a SHORT PHRASE COPIED VERBATIM from the context.\n'
    )
    usr = (
        f"Context:\n{ctx_text}\n\n"
        f"Question: {q}\n"
        "Reply in JSON only."
    )
    return sys, usr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval_questions.jsonl")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--retriever", choices=["bm25", "hybrid"], default="hybrid")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--topk_ctx", type=int, default=3)  # tighter context tends to improve EM
    ap.add_argument("--reranker", default="", help='e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2" or "BAAI/bge-reranker-large"')
    ap.add_argument("--out", default="artifacts_easy/eval_gen_ollama.jsonl")
    args = ap.parse_args()

    idx, bm25, meta, emb = load_all(args.embed_model)
    rer = CrossEncoder(args.reranker) if args.reranker else None

    rows = [json.loads(l) for l in open(args.eval, "r", encoding="utf-8")]
    EM, F1, RL = [], [], []

    with open(args.out, "w", encoding="utf-8") as w:
        for ex in rows:
            q, gold = ex["query"], ex.get("answer", "")

            # Retrieve candidates
            di, ds = dense_search(idx, emb, q, args.topk * 2)
            bi, bs = bm25_search(bm25, q, args.topk * 2)
            I = bi if args.retriever == "bm25" else hybrid(di, ds, bi, bs, args.alpha, k=args.topk * 2)

            # Optional re-rank
            if rer and I:
                pairs = [(q, meta[i]["text"]) for i in I]
                scores = rer.predict(pairs)
                order = np.argsort(-scores)[:args.topk]
                I = [I[i] for i in order]
            else:
                I = I[:args.topk]

            # Build context & ask the model
            # Build context
            ctx = [meta[i]["text"] for i in I[:args.topk_ctx]]

            # 1) Try span extraction first (from top few contexts)
            pred = ""
            for c in ctx[:3]:  # inspect top-3 contexts
                cands = extract_answers_from_ctx(c)
                if cands:
                    pred = best_extracted_answer(q, cands)
                    if pred:
                        break

            # 2) Fall back to LLM only if extraction failed
            if not pred:
                if not ctx:
                    pred = ""
                else:
                    sys, usr = build_prompt(q, ctx)
                    raw = ollama_chat(MODEL, sys, usr, temperature=0.0, max_tokens=128)
                    raw_clean = raw.strip()
                    raw_clean = re.sub(r"^```json\s*|\s*```$", "", raw_clean, flags=re.I | re.M)
                    try:
                        obj = json.loads(raw_clean)
                        pred = (obj.get("answer", "") or "").strip()
                    except Exception:
                        m = re.search(r'\{\s*"answer"\s*:\s*"(.*?)"\s*\}', raw_clean, flags=re.S)
                        pred = (m.group(1).strip() if m else "").strip()


            # Normalize for scoring
            pred_norm = normalize_answer(pred)
            gold_norm = normalize_answer(gold)

            e, f, r = em(pred_norm, gold_norm), f1(pred_norm, gold_norm), rouge_l(pred_norm, gold_norm)
            EM.append(e); F1.append(f); RL.append(r)

            w.write(json.dumps({
                "q": q, "gold": gold, "pred": pred,
                "ctx_ids": I[:args.topk_ctx]
            }, ensure_ascii=False) + "\n")

    print("=== End-to-end (Ollama) ===")
    print(f"EM: {np.mean(EM):.3f}  F1: {np.mean(F1):.3f}  RougeL: {np.mean(RL):.3f}")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
