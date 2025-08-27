"""
augment_from_sheet.py

Generate FAQ Q&A pairs from your Excel sheet using an Ollama model.
Saves JSONL at --out (default: data/faq_raw.jsonl).

Usage (Mac/Linux):
  python augment_from_sheet.py \
    --excel "requested models list (3).xlsx" \
    --model "llama3.1:8b" \
    --per_model 3 --sample 10 --out "data/faq_raw.jsonl"

Requires: pandas, openpyxl, requests  (pip install pandas openpyxl requests)
Ollama:   https://localhost:11434 (Ollama app running)
"""

import argparse, os, time, json, re
import pandas as pd
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SYS_PROMPT = """You are a precise data generator for enterprise FAQ datasets.
Given a target model name and optional link/title, produce realistic FAQ pairs.
Rules:
- Output STRICT JSON: {"pairs":[{"question": "...","answer":"..."}]}
- Up to {n_pairs} pairs. Short, accurate answers (1–4 sentences).
- If unsure, stay generic; avoid made-up version numbers or prices.
- If the target is an embedding/reranker/TTS or non-chat model, generate FAQs about usage, input/output formats, 
typical applications, evaluation metrics, latency/throughput, integration steps 
(e.g., with llama.cpp/Ollama), and limitations, without inventing version numbers.
"""

USER_TMPL = """Target model: "{model_name}"
Reference link (optional): {url}
Title (optional): {title}
Generate up to {n_pairs} FAQs about internal company usage (evaluation, deployment, quantization, GPU needs, latency/throughput, compatibility with llama.cpp/Ollama, etc.)."""

def pick_columns(df: pd.DataFrame):
    cm = next((c for c in df.columns if "model" in str(c).lower() and "name" in str(c).lower()), None)
    cu = next((c for c in df.columns if str(c).strip().lower() in ("url","issue url","link")), None)
    ct = next((c for c in df.columns if "title" in str(c).lower()), None)
    if cm is None: cm = df.columns[0]
    return cm, ct, cu

def read_rows(excel_path: str):
    df = pd.read_excel(excel_path)
    cm, ct, cu = pick_columns(df)
    rows = []
    for _, r in df.iterrows():
        m = str(r.get(cm, "")).strip()
        if not m or m.lower() == "nan": continue
        rows.append({
            "model_name": m,
            "title": str(r.get(ct, "")).strip() if ct else "",
            "url": str(r.get(cu, "")).strip() if cu else "",
        })
    return rows

def call_ollama(model: str, system: str, user: str, temperature=0.1, retries=2):
    import requests, time, os, json

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    chat_url = f"{OLLAMA_HOST}/api/chat"
    gen_url  = f"{OLLAMA_HOST}/api/generate"

    def _chat(sysmsg, usrmsg):
        body = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": sysmsg},
                {"role": "user",   "content": usrmsg},
            ],
            "options": {"temperature": temperature, "format": "json"},
            "stream": False,
        }
        r = requests.post(chat_url, json=body, timeout=180)
        # If server doesn't support /api/chat, 404 here
        if r.status_code == 404:
            raise RuntimeError("CHAT_NOT_SUPPORTED")
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _generate(sysmsg, usrmsg):
        # Emulate chat by concatenating system + user
        prompt = (
            f"{sysmsg.strip()}\n\n"
            f"USER:\n{usrmsg.strip()}\n\n"
            "ASSISTANT:"
        )
        body = {
            "model": model.strip(),
            "prompt": prompt,
            "format": "json",               # ask for strict JSON
            "options": {"temperature": temperature},
            "stream": False,
        }
        r = requests.post(gen_url, json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        # /api/generate returns { "response": "...", "done": true, ... }
        return data.get("response", "")

    last_err = None
    for i in range(retries):
        try:
            try:
                return _chat(system, user)
            except RuntimeError as e:
                if str(e) == "CHAT_NOT_SUPPORTED":
                    # fallback to /api/generate
                    return _generate(system, user)
                raise
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err


        

def parse_pairs(txt: str):
    import json, re
    cleaned = txt.strip()
    # remove code fences if present
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.I | re.M)

    # try direct JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("pairs"), list):
            return [{"question": p.get("question","").strip(),
                     "answer": p.get("answer","").strip()}
                    for p in obj["pairs"] if p.get("question") and p.get("answer")]
    except Exception:
        pass

    # fallback: find a JSON object containing "pairs"
    m = re.search(r"\{.*?\"pairs\"\s*:\s*\[.*?\]\s*\}", cleaned, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("pairs"), list):
                return [{"question": p.get("question","").strip(),
                         "answer": p.get("answer","").strip()}
                        for p in obj["pairs"] if p.get("question") and p.get("answer")]
        except Exception:
            pass
    return []



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to your Excel file")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--per_model", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0, help="If >0, only first N rows")
    ap.add_argument("--out", default="data/faq_raw.jsonl")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--only", default="", help="Only process rows whose Model Name contains this text")
    ap.add_argument("--model_fallback", default="", help="Optional fallback model if the first fails")

    args = ap.parse_args()

    rows = read_rows(args.excel)
    if args.sample > 0:
        rows = rows[:args.sample]
    
    if args.only:
        part = args.only.lower()
        rows = [r for r in rows if part in r["model_name"].lower()]


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_total, n_ok = 0, 0

    with open(args.out, "w", encoding="utf-8") as w:
        for row in rows:
            n_total += 1
            sys = SYS_PROMPT.replace("{n_pairs}", str(args.per_model))
            user = USER_TMPL.format(
                model_name=row["model_name"],
                url=row["url"] or "N/A",
                title=row["title"] or "N/A",
                n_pairs=args.per_model
            )
            txt = call_ollama(args.model.strip(), sys, user, temperature=args.temperature)
            pairs = parse_pairs(txt)
            if not pairs and args.model_fallback:
                txt2 = call_ollama(args.model_fallback.strip(), sys, user, temperature=args.temperature)
                pairs = parse_pairs(txt2)

            for qa in pairs:
                w.write(json.dumps({"question": qa["question"], "answer": qa["answer"], "model_name": row["model_name"]}, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Done. Rows processed: {n_total}, with generated FAQs: {n_ok}. Output -> {args.out}")

if __name__ == "__main__":
    main()
