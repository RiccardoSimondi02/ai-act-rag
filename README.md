# ai-act-rag

Local, privacy-first RAG over the EU AI Act: structure-aware chunking, dense retrieval with a reranker, cited answers checked by code, an LLM judge validated against hand labels, and an evaluation gate in CI. Everything, evaluation included, runs on one GPU.

The generator runs locally through Ollama for one reason: the target use case involves confidential documents. 



```mermaid
flowchart LR
  Q[Question] --> E[Dense retrieval<br/>Qwen3-Embedding-0.6B, top 30]
  E --> R[Reranker<br/>bge-reranker-v2-m3]
  R --> C[Context: first 1,000 words]
  C --> G[Gemma 3 12B via Ollama<br/>JSON: claims + sources, or abstain]
  G --> V[Citation checks in code] --> A[Answer with links to EUR-Lex]
```

## Corpus

- EU AI Act, **consolidated English text of 27 July 2026** (CELEX `02024R1689-20260727`), which includes the amendments of the Digital Omnibus on AI (Regulation (EU) 2026/1744).
- Raw HTML downloaded from [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02024R1689-20260727) and stored unchanged as `data/raw/ai_act.html` (SHA-256 `391a9395f0933cb3735c160d176a9fdc6db735dac6a05b1f129e2a6113a59b7f`).

## Ingestion

`scripts/build_units.py` turns the HTML into `data/processed/units.jsonl`, emitting one record for every citable unit.

| Source | Unit | Id | Citation |
|---|---|---|---|
| Article with numbered paragraphs | paragraph (with its unnumbered subparagraphs) | `art_99.par_3` | Article 99(3) |
| Article 3 | one definition | `art_3.pt_12` | Article 3(12) |
| Article without numbered paragraphs | whole article | `art_16` | Article 16 |
| Annex ≤ 500 words | whole annex | `anx_II` | Annex II |
| Longer annex | section or first-level point | `anx_III.pt_1` | Annex III, point 1 |

Every record also carries the parent title, the sentence that opens a list of points (`parent_intro`), the CELEX of the act in force at the start of the text (`act_at_start`), and a link back to the article on EUR-Lex.

Because ids are built from structure rather than from a running counter, the evaluation dataset can keep pointing at them even when the parser is modified.

Several decisions shaped the parser. 
- Articles 102 through 110 are skipped, since they only modify other EU acts and would pollute retrieval. 
- Recitals are absent too, because EUR-Lex consolidated texts leave out the preamble. 
- Footnotes and amendment markers such as ▼B and ▼M1 are removed from the body text and stored as metadata instead. 
- The title of Article 1 has a typo in the source ("Subject matter'"), and the parser corrects it there rather than editing the raw file.

## Evaluation dataset

`eval/dataset/questions.jsonl` contains 60 questions, hand-curated and verified against the source. Every question comes with an expected answer plus the ids of the units holding it (`gold_units`).

| Type | Count | What it tests |
|---|---|---|
| factual | 22 | answer in one unit, often worded differently from the text ("chatbot", "loan") |
| exact_term | 16 | definitions and questions that cite a provision ("Article 26(1)", "Annex XIII") |
| multi_hop | 12 | answer needs two units |
| abstain | 10 | plausible questions the corpus cannot answer; 3 of them have a related unit the answer may point to, but must not present as the answer |

Labels live on units, not on chunks, so every chunking strategy is measured against the same ground truth.

The dev/test split is 50/50, stratified by type, with a fixed seed (`scripts/assign_split.py`).

## Retrieval

Metrics are computed over unit ids: recall@k and MRR, with abstain questions left out. The primary criterion is recall@20, because retrieval is only the first of two stages and the reranker will shuffle the candidates anyway. What counts here is whether the answer makes it into the candidate pool.

| dev, 25 questions | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|
| BM25 | 0.56 | 0.74 | 0.80 | 0.51 |
| Dense (Qwen3-Embedding-0.6B) | 0.68 | 0.90 | 0.90 | 0.71 |
| Hybrid (RRF, k=60) | 0.80 | 0.82 | 0.90 | 0.69 |

**Experiments** (dev, recall@20 / MRR; one change at a time)

| Change | Result | Outcome |
|---|---|---|
| BM25 baseline: lowercase word tokens, unit text only | 0.64 / 0.39 | baseline |
| + Snowball stemming ("fined"/"fines", "prohibited"/"prohibition") | 0.72 / 0.48 | kept |
| + citation in the chunk text ("Article 99(3) - …") | 0.80 / 0.51 | kept: questions citing a provision jump from >100 to the top 5 |
| + article title in the chunk text | 0.80 / 0.48 | dropped: the title boosts every paragraph of the article equally, 7 regressions |
| stop-word removal (scikit-learn list) | 0.78 / 0.47 | dropped: the generic list removes "system", "not", "except", "third" |
| Lucene IDF (bm25s) instead of rank_bm25's IDF floor | 0.80 / 0.47 | not adopted: no measurable gain |
| Dense retriever | 0.90 / 0.71 | solves vocabulary mismatch ("chatbot", "outside the EU" vs "third countries") |
| Hybrid RRF of BM25 + dense | 0.90 / 0.69 | tie on recall@20; see Reranking for the final choice |

**A few observations**. 
- Dense and BM25 fail on entirely different questions: dense gets tripped up by identifiers such as "Article 4a" or "Annex XIII", while BM25 gets tripped up by paraphrases. 
- RRF rewards agreement between the two, so a chunk that one retriever ranks first but the other never returns scores 1/61, whereas a chunk that both place at rank 20 scores 2/80 and wins the slot. That explains why the hybrid improves on identifiers (recall@5 climbs from 0.68 to 0.80) yet sheds some of the paraphrase questions the dense retriever had already handled. 
- First-stage scores cannot make the abstention call by themselves: the top score for unanswerable questions sits in the same band as for answerable ones, under both BM25 and dense. Reranker scores offer a partial signal, discussed in Reranking. 
- The misses that remained were long units, such as Article 5(1) at 877 words, which the Chunking section deals with.

**Architectural choices** 
- there is no vector database, because 673 embeddings fit comfortably in memory and exact search amounts to a single matrix product. Approximate indexes like FAISS or HNSW only pay off with millions of vectors; here they would introduce a service and buy nothing. 
- The embedding model is Qwen3-Embedding-0.6B. Queries are encoded with the model's query prompt, documents without one. Embeddings are cached on disk together with the model name and chunk ids, and get recomputed whenever either changes. 

## Chunking

Chunks of differing sizes cannot be fairly compared at the same k, since 20 windows of 200 words carry three times the text of 20 units. Strategies are therefore compared under an equal text budget: recall within the first 1,000 and 2,000 words retrieved, which is the amount of text the reranker and generator actually get to see.

| dev, dense retriever | recall@1000w | recall@2000w | MRR |
|---|---|---|---|
| A. Structural: one chunk per unit, with citation | 0.84 | 0.90 | 0.71 |
| B. Fixed: 200-word windows, 50 overlap, no citation | 0.74 | 0.80 | 0.68 |
| C. Structural, units over 200 words split into windows that keep the citation | 0.94 | 0.94 | 0.72 |

A and B break on different questions. B repairs long units, where a single embedding averages ten distinct provisions, but it damages short ones by blending them with their neighbours and it loses identifiers. C preserves the structure for the 627 short units and splits the 46 long ones, which make up 29% of the text. It is the standard structure-aware recipe with a size cap.

For fixed windows, a unit is counted as contained when the window covers at least half of it or at least 50 of its words. Under an equal budget, the dense retriever overtakes the hybrid (0.94 versus 0.86 at 1,000 words with C).

## Reranking

A cross-encoder (`BAAI/bge-reranker-v2-m3`, max length 512 tokens) reorders the top 30 candidates. In contrast to the bi-encoder, it reads the question and the chunk side by side, which buys precision but also means it must run over every pair at query time. Hence two stages.

| dev, chunking C | recall@5 | recall@1000w | recall@20 | MRR | mean latency |
|---|---|---|---|---|---|
| Dense | 0.76 | 0.94 | 0.94 | 0.72 | 22 ms |
| **Dense + reranker (final)** | 0.90 | 0.92 | 0.96 | 0.89 | 258 ms |
| Hybrid RRF | 0.86 | 0.86 | 0.90 | 0.67 | 23 ms |
| Hybrid RRF + reranker | 0.86 | 0.88 | 0.92 | 0.85 | 262 ms |

The final pipeline is chunking C, then dense retrieval, then the reranker over the top 30. It equals or beats the hybrid on every metric while using one fewer component. Once the reranker reads the text, BM25's edge on identifiers is no longer required, yet RRF can still drop a unit that the dense retriever found on its own (q044 lands 38th after fusion, outside the candidates). The hybrid remains implemented and selectable through configuration, since on corpora packed with identifiers (case or contract numbers) it may prove the better fit. The gap on this dev set is a single question, so the choice leans on equal-or-better results plus simplicity.




## Generation

Gemma 3 12B runs locally through Ollama, at temperature 0, with a fixed seed and an 8,192-token context. The context consists of the reranked chunks up to 1,000 words, numbered. The answer must obey a JSON schema that Ollama enforces: `{"abstain": bool, "claims": [{"text", "sources": [numbers]}]}`.

Three checks run on each answer, none of them involving a model. Every cited source must exist in the context, every claim must cite at least one source, and the answer must cite at least one gold unit.

| dev, 30 questions | correct abstentions (of 5) | invented sources | uncited claims | cites a gold unit (of 25) | generation p50 / p95 |
|---|---|---|---|---|---|
| No context (same model) | 0 | – | – | – | 2.0 / 3.6 s |
| Retrieval, 1,000 words | 3 | 0 | 0 | 23 | 2.4 / 5.6 s |

- Without any context the model never abstains: it answers all five unanswerable questions. 
- A 1,000-word budget outperforms 2,000. The gold unit appears in the context for 24 of 25 questions under both settings.
- When retrieval fails to surface the answer (q058, "Article 4a"), the model abstains rather than inventing one. That is the behaviour we want.
- None of the real errors are caught by the checks, since they are all semantic in nature.

## Evaluating the answers

A second local model grades the answers: Qwen 2.5 14B, chosen from a different family than the generator so it does not mark its own homework. Each criterion has its own prompt and JSON schema, with the model reasoning before it commits to a verdict, using the same definitions as the labelling guidelines (`eval/labels/guidelines.md`).

The judge was validated against hand labels on the dev split, covering 40 answers and 40 randomly sampled claims, all annotated before its verdicts were seen.

| Criterion | Cases | Agreement | Cohen's κ | Use |
|---|---|---|---|---|
| Claim supported by its cited sources | 40 | 92.5% | 0.68 | judge |
| Correctness vs expected answer | 32 | 66–69% | 0.46–0.51 | judge, large differences only |
| Off-topic claims | 36 | 61% | 0.32 | hand labels only |
| Abstention on unanswerable questions | 7 | 86% | – | hand labels only |

Claim-level judgements hold up well; holistic grades do not. On correctness the judge marks 5 answers as partial that human labels called correct, consistently citing "extra details", even though the prompt states that extra true claims must not lower the grade. This is the inverse of the usual verbosity bias. It labels the exceptions and definitions that the rule excludes as off topic, and on abstention it misses the one case that actually matters.

The judge is not deterministic. Judged four times, the same answers scored anywhere from 9 to 12 correct out of 24 and 83.5 to 86.8% supported claims (κ on correctness between 0.46 and 0.51). All of the system's run-to-run noise traces back to the judge.

| dev | Correct / partial / wrong (judge) | Claims supported (judge) | Answers with off-topic claims (hand) |
|---|---|---|---|
| No context | 1 / 16 / 8 (one run) | – | – |
| Retrieval | 9–12 / 10–13 / 2–3 | 84–87% | 7 / 26 |

The gap is wide enough to hold up against this noise, but differences of three or four answers or fewer between configurations cannot be trusted to this judge. Its agreement is measured once more on test answers it has never seen, at the final evaluation.

## CI

GitHub Actions (`.github/workflows/ci.yml`) fires on every push and pull request. The `test` job runs the unit tests covering the parser and coverage, the dataset, the metrics, and the citation checks. The `eval-gate` job rebuilds the units, runs the final retrieval pipeline on the dev split on CPU, and fails whenever a metric drops below `eval/thresholds.json` (the CPU value minus one question: recall@1000w 0.88, MRR 0.853). Models and embeddings are cached between runs.

Generation and the judge stay out of CI. They depend on Ollama and a GPU, and the judge's own noise of roughly three answers would make the gate fail at random.

## Service

`docker compose up` brings up two containers on the GPU: Ollama serving Gemma 3 12B, and a FastAPI app (`POST /ask`, `GET /health`, plus a minimal page on `/` with clickable citations pointing to EUR-Lex). Models load once at startup, followed by a warm-up question. Ollama keeps the generator resident in memory, and requests are handled one at a time, since all models share a single GPU.

Every request emits one JSON log line containing timings per stage, tokens, abstention, number of sources, top reranker score, and the citation checks. The question text itself is never logged.

| 30 dev questions through the API | p50 | p95 | share of time |
|---|---|---|---|
| Retrieval | 0.09 s | 0.31 s | 3% |
| Reranker | 0.66 s | 0.99 s | 16% |
| Prompt processing | 0.45 s | 0.51 s | 11% |
| Generation | 1.91 s | 5.87 s | 53% |
| Other (context selection, citation cleaning, Ollama overhead) | 0.65 s | 1.19 s | 16% |
| Total | 3.86 s | 7.97 s | |

## Final evaluation (test split)

| | dev | test |
|---|---|---|
| Retrieval recall@1000w / MRR | 0.92 / 0.89 | 0.90 / 0.82 |
| Correct / partial / wrong, with retrieval (judge) | 9–12 / 10–13 / 2–3 | 14 / 10 / 1 |
| Correct / partial / wrong, no context (judge) | 1 / 16 / 8 | 3 / 15 / 7 |
| Claims supported | 84–87% | 92% |
| Unanswerable questions correctly declined (hand) | 3 of 5 (4 via service) | 1 of 5 |

Abstention does not carry over to unseen data. On test the system answers 4 of 5 unanswerable questions using adjacent provisions.

Retrieval falls slightly, which is expected on questions it has not seen. One of the three misses is an identifier ("Article 6(1)", rank 40, outside the reranker's candidates), precisely the scenario the hybrid retriever is built for. That is an argument for the hybrid that the dev set never surfaced.

The judge repeats its dev behaviour on unseen data. On correctness it marks 4 answers partial that human labels called correct and never errs the other way (κ 0.37; hand labels say 9 of 11 correct, the judge says 5), which makes its correctness figures a lower bound. On support it agrees 95% (the κ is 0 only because nearly all labels are "supported"). On abstention it agrees on just 1 of 5, which is why abstention is always hand-labelled.


## Limitations

- The system seldom declines unanswerable questions once related provisions are retrieved (1 of 5 on test).
- The LLM judge is only moderately reliable on correctness and skews towards "partial". Abstention and off-topic claims are hand-labelled.
- `act_at_start` captures the act in force where a unit begins; amendments that occur inside a unit are not recorded.
- Questions stay within what the text literally says, not what it legally implies.
- The dev set is small (25 answerable questions), so differences of one or two questions are not conclusive in isolation.
- Only one generator (Gemma 3 12B) was evaluated. Comparing local generators is left for later work.
- The CI gate's failure path is not demonstrated in this repository.
- A prompt rule tells the model to keep source numbers out of the claim text, yet a few claims still contain them ("[1]"). This is cosmetic, since citations are read from the `sources` field.

## Observation
A larger generator (local, or an API model where privacy allows) may decline unanswerable questions more reliably; untested, and at odds with the local-only design.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/build_units.py
pytest
python scripts/eval_retrieval.py dense_qwen3_v1       # any run name in scripts/eval_retrieval.py
python scripts/compare_runs.py bm25_stem_v1 dense_qwen3_v1

# Service (needs Docker with GPU support)
docker compose up                                   # then open http://localhost:8000
python scripts/load_test.py                         # dev questions through the API, timings from the logs
```
