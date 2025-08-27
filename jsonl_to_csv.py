import json, csv, sys, os
src = sys.argv[1] if len(sys.argv) > 1 else "data/faq_raw.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/faq_raw.csv"
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
with open(src,"r",encoding="utf-8") as f, open(dst,"w",newline="",encoding="utf-8") as g:
    w = csv.writer(g); w.writerow(["Model Name","Question","Answer"])
    for line in f:
        o = json.loads(line)
        w.writerow([o.get("model_name",""), o.get("question",""), o.get("answer","")])
print("Wrote", dst)
