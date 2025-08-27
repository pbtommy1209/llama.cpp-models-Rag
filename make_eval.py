import json, os, random, sys

src = sys.argv[1] if len(sys.argv) > 1 else "data/faq_raw.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/eval_questions.jsonl"
sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50

rows = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        try:
            o = json.loads(line)
            q = (o.get("question","") or "").strip()
            a = (o.get("answer","") or "").strip()
            if q and a:
                rows.append({"query": q, "answer": a})
        except:
            pass

if not rows:
    raise SystemExit(f"No Q&A found in {src}")

# dedup by question
seen, dedup = set(), []
for r in rows:
    k = r["query"].lower()
    if k not in seen:
        seen.add(k)
        dedup.append(r)

random.seed(42)
sample = dedup if len(dedup) <= sample_size else random.sample(dedup, sample_size)

os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
with open(dst, "w", encoding="utf-8") as w:
    for r in sample:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {dst} with {len(sample)} rows")
