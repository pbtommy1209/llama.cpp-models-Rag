<!-- ERRSEC START -->

##  Error Case Analysis & Improvement Plan

###  Sample error cases (excerpt)

| # | Category | Question | Gold | Pred | Ctx[0] (first 140 chars) |
|---:|---|---|---|---|---|
| 1 | other | What is the typical use case for Pixtral-12b-240910 in an enterprise setting? | Pixtral-12b-240910 is designed to be used as a visual model for image generation tasks, such as generating product images or creating visual content for marketing campaigns. | Pixtral-12b-240910 is designed to be used as a visual model for image generation tasks, such as generating product images or creating visual content for marketing campaigns. | Q: What is the typical use case for Pixtral-12b-240910 in an enterprise setting? A: Pixtral-12b-240910 is designed to be used as a visual mo... |
| 2 | other | How does Pixtral-12b-240910 perform on deployment in a production environment? | Pixtral-12b-240910 has been shown to achieve high accuracy and efficiency in deployment, making it suitable for real-time applications such as product recommendation or image classification. | Pixtral-12b-240910 has been shown to achieve high accuracy and efficiency in deployment, making it suitable for real-time applications such as product recommendation or image classification. | Q: How does Pixtral-12b-240910 perform on deployment in a production environment? A: Pixtral-12b-240910 has been shown to achieve high accur... |
| 3 | other | Can Pixtral-12b-240910 be used with other models like llama.cpp/Ollama? | Yes, Pixtral-12b-240910 is compatible with llama.cpp/Ollama and can be integrated into existing workflows for seamless visual model usage. | Yes, Pixtral-12b-240910 is compatible with llama.cpp/Ollama and can be integrated into existing workflows for seamless visual model usage. | Q: Can Pixtral-12b-240910 be used with other models like llama.cpp/Ollama? A: Yes, Pixtral-12b-240910 is compatible with llama.cpp/Ollama an... |
| 4 | other | What are the latency/throughput expectations for Pixtral-12b-240910 in a GPU-accelerated environment? | Pixtral-12b-240910 is optimized for GPU acceleration, resulting in fast inference times and high throughput, making it suitable for applications requiring rapid visual model processing. | Pixtral-12b-240910 is optimized for GPU acceleration, resulting in fast inference times and high throughput, making it suitable for applications requiring rapid visual model processing. | Q: What are the latency/throughput expectations for Pixtral-12b-240910 in a GPU-accelerated environment? A: Pixtral-12b-240910 is optimized ... |
| 5 | other | How does Pixtral-12b-240910 handle quantization for deployment on edge devices? | Pixtral-12b-240910 supports quantization, allowing for efficient deployment on edge devices while maintaining acceptable accuracy levels. | Pixtral-12b-240910 supports quantization, allowing for efficient deployment on edge devices while maintaining acceptable accuracy levels. | Q: How does Pixtral-12b-240910 handle quantization for deployment on edge devices? A: Pixtral-12b-240910 supports quantization, allowing for... |

> Note: Some entries appear correct (Gold ≈ Pred). Such cases likely surfaced due to strict matching or normalization differences during evaluation.

###  Category summary

`Category counts: {'other': 50}`

**Interpretation.**
- A large “other” bucket usually means our auto-categorizer couldn’t confidently label the failure as *retrieval_miss*, *extraction_miss*, or *paraphrase_drift*.  
- When Gold and Pred look identical but are still flagged, the cause is typically **scoring/normalization** (e.g., whitespace, punctuation, case, or articles) rather than a true model error.

###  Root causes observed

- **Scoring false negatives.** Normalization differences (e.g., punctuation, spacing, “the/a/an”) can mark correct answers as wrong.
- **Over-broad “other” class.** Our classifier doesn’t split formatting issues vs. genuine content mismatches.
- **Occasional paraphrasing.** When the reader generates rather than copies a span, EM/Rouge can drop even if the answer is semantically correct.

###  Improvement plan

**A. Evaluation & labeling**
- **Harden normalization** in scoring (lowercase, trim, collapse whitespace, strip punctuation and articles) to avoid false negatives.
- **Tighten error labels** to break “other” into:
  - `format_mismatch` (answer equals after normalization),
  - `near_duplicate` (minor token differences),
  - `true_mismatch` (semantic/content difference).

**B. Reader/answering**
- **Extractor-first, LLM-fallback** (already enabled): keep copying the `A:` span verbatim; only call the LLM if no span is found.
- **Context size**: keep **tight context** (`--topk_ctx 2–3`) to reduce paraphrasing drift.
- **Temperature**: keep **`temperature=0.0`** and strict JSON to enforce verbatim answers.

**C. Retrieval**
- **Candidate pool**: set `--topk 50` to reduce misses; keep re-ranking on (e.g., `BAAI/bge-reranker-base` for speed).
- **Hybrid for OOV terms**: prefer **hybrid (dense+BM25)** when names are unusual, otherwise BM25+rereanker is fine for FAQ-style data.

**D. Indexing / preprocessing**
- **Q/A-intact chunks**: keep each FAQ pair as a single chunk:  
  `Q: …` on one line + `A: …` on the next line.  
  Avoid splitting that across chunks.
- **Glossary for aliases/OOV** (optional): attach a short “Aliases:” line to each chunk (e.g., `phi4`, `phi-4`, `phi four`) to boost both BM25 and dense hits.
- **Chunk boundaries**: if any `A:` is cut by chunking, reduce `--max_tokens` (e.g., 256) and/or increase `--overlap` (e.g., 64).

**E. Safety & citations**
- **Hallucination guard**: if the answer cannot be copied from context, return an empty string.  
- **Traceability**: keep `ctx_ids` in outputs so answers can be traced to the exact snippets used.

**F. Quick verification loop (recommended)**
1) Re-run evaluation with hardened normalization.  
2) If “other” remains high, print 10 random “other” rows to confirm whether they’re formatting-only; if yes, they’re not model errors.  
3) For remaining true mismatches, raise `--topk` and verify `A:` isn’t cut.

<!-- ERRSEC END -->
