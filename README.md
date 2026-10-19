# RAG Assignment – Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two-stage retrieval over **The Bharatiya Nyaya Sanhita, 2023** (India's criminal code, 102 pages, 358 sections).

| | |
|---|---|
| Notebook | [`RAG_Assignment.ipynb`](RAG_Assignment.ipynb) (all code + outputs) |
| Report | [`report.pdf`](report.pdf) |
| Document | [`data/Bharatiya_Nyaya_Sanhita_2023.pdf`](data/Bharatiya_Nyaya_Sanhita_2023.pdf) – official Gazette copy (mha.gov.in) |
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine) |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `openai/gpt-oss-120b` via the Groq API, temperature 0 |

**Pipeline 1:** bi-encoder Top 10 → Top 3 → LLM → Answer 1
**Pipeline 2:** same Top 10 → cross-encoder re-rank → Top 3 → LLM (identical prompt) → Answer 2

## How to run
```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env     # free key from https://console.groq.com (.env is git-ignored)
jupyter notebook RAG_Assignment.ipynb
```
The bi-encoder, FAISS index and cross-encoder run locally on CPU; only the answer generation uses the API.
LLM answers are cached in `llm_cache.json`, so re-running the notebook reproduces the exact answers discussed in the report (delete the file to query the API again).

## Results (8 stress-test queries)
| | Hit@1 | Hit@3 | MRR@10 | Correct answers |
|---|---|---|---|---|
| Pipeline 1 (Bi-Encoder) | 0.000 | 0.125 | 0.235 | 1 / 8 |
| Pipeline 2 (Re-ranked) | 0.625 | 0.750 | 0.719 | 5 / 8 |

Retrieval latency: ~6 ms (bi-encoder) vs ~227 ms (bi-encoder + cross-encoder on 10 pairs). Charts are in [`figures/`](figures/).
