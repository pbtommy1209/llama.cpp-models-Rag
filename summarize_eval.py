
import json, sys, numpy as np
def load(path):
    rows=[json.loads(l) for l in open(path,"r",encoding="utf-8")]
    EM=[];F1=[];RL=[]
    for r in rows:
        p=r.get("pred",""); g=r.get("gold","")
        def tok(s):
            import re; return re.findall(r"[A-Za-z0-9_]+", s.lower())
        def f1(p,g):
            from collections import Counter
            P,G=Counter(tok(p)),Counter(tok(g))
            common=sum((P&G).values())
            if not p and not g: return 1.0
            if not p or not g or common==0: return 0.0
            prec=common/len(tok(p)); rec=common/len(tok(g))
            return 2*prec*rec/(prec+rec)
        def em(p,g): return int(p.strip()==g.strip())
        def rouge_l(p,g):
            from difflib import SequenceMatcher
            s=SequenceMatcher(None,p,g)
            return sum(m.size for m in s.get_matching_blocks())/max(1,len(g))
        EM.append(em(p,g)); F1.append(f1(p,g)); RL.append(rouge_l(p,g))
    return float(np.mean(EM)), float(np.mean(F1)), float(np.mean(RL))

base = sys.argv[1]  
final= sys.argv[2]  
bem,bf1,brl = load(base)
fem,ff1,frl = load(final)
print("| System | EM | F1 | Rouge-L |")
print("|---|---:|---:|---:|")
print(f"| Baseline (BM25) | {bem:.3f} | {bf1:.3f} | {brl:.3f} |")
print(f"| Final (Hybrid+Rerank) | {fem:.3f} | {ff1:.3f} | {frl:.3f} |")
