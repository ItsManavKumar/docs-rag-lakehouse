# docs-rag-lakehouse

A retrieval-augmented generation (RAG) pipeline on **Databricks** that answers questions over a folder of PDFs
and cites the document and page for every fact. It's built as a **medallion Lakehouse** (Bronze → Silver → Gold Delta
tables) with **PySpark**, and evaluated with a question set whose answers were checked by hand.

It works with any set of documents: `config/config.yaml` points at a folder of PDFs, so the corpus can be swapped
without code changes.

> Independent portfolio project built over **publicly available** product data sheets. Not affiliated with,
> endorsed by, or built for DuluxGroup.

---

## Architecture

```mermaid
flowchart LR
    A[PDFs in a UC Volume<br/>+ manifest.csv] -->|binaryFile + mapInPandas| B[(Bronze<br/>bronze_pages<br/>1 row per page)]
    B -->|clean · dedupe · section-aware chunking<br/>applyInPandas + window fns| C[(Silver<br/>silver_chunks<br/>tagged chunks)]
    C -->|bge-small-en-v1.5| D[(Gold<br/>gold_chunk_embeddings<br/>Delta + CDF)]
    D --> E[FAISS index<br/>or Databricks Vector Search]
    Q[Question] --> F{Product or brand<br/>named?}
    F -->|yes: metadata filter| E
    F -->|no| E
    E -->|top-k chunks| G[Claude Sonnet 5<br/>via Vercel AI Gateway]
    G --> H[Answer + citations<br/>or "not in the documents"]
    H --> L[(qa_log)]
    C -.-> DQ[(silver_dq_checks)]
    H -.-> EV[(eval_results)]
```

| Layer | Table | What it holds |
|---|---|---|
| Bronze | `bronze_pages` | one row per PDF page: raw text (tables rendered as `key: value`), file, brand, product, doc type, source URL, `needs_ocr` flag |
| Silver | `silver_chunks` | cleaned, de-duplicated, section-aware chunks tagged with brand / product / doc type / section / page range |
| Silver | `silver_dq_checks` | data-quality report (OCR-needed pages, unknown brands, empty or oversized chunks, duplicate IDs) |
| Gold | `gold_chunk_embeddings` | chunk metadata + 384-dim embedding; Change Data Feed on so it can back a Vector Search index |
| — | `qa_log`, `eval_results` | every question/answer with citations and retrieved chunks; evaluation runs |

## Repo layout

```
config/config.yaml        corpus folder, chunking, model, LLM endpoint (no secrets)
notebooks/                Databricks notebooks (source format), run in order
  00_setup_and_checks     install, create schema/volumes, check model + LLM + secret access
  01_bronze_ingest        PDFs -> bronze_pages
  02_silver_chunks        clean, dedupe, chunk, DQ checks -> silver_chunks
  03_gold_embeddings      embeddings -> gold_chunk_embeddings + FAISS index
  04_rag_query            retrieval + grounded answers with citations
  05_evaluation           hit rate@k, answer accuracy, results table
src/docs_rag/             plain-Python package used by notebooks, Spark UDFs and local runs
scripts/                  run_local.py (whole pipeline without Spark), download_pdfs.py,
                          download_model.py, draft_eval_questions.py
eval/questions.jsonl      evaluation questions (answers checked against the PDFs)
tests/                    unit tests + a local-Spark test of Bronze/Silver on synthetic PDFs
```

## How to run

### 1. Get the documents (laptop)
1. Collect 30–50 public data sheets and list them in `data/manifest.csv` (`file,brand,product,doc_type,source_url`;
   see `data/manifest.example.csv`). Use official sources only.
2. `pip install -r requirements.txt && python scripts/download_pdfs.py data/manifest.csv data/pdfs`

### 2. Databricks Free Edition
1. **Workspace → Create → Git folder** → paste this repo's URL.
2. Store the API key as a secret (from your laptop, with the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/)):
   ```bash
   databricks secrets create-scope docs-rag
   databricks secrets put-secret docs-rag ai-gateway-key     # paste the Vercel AI Gateway key when prompted
   ```
3. Run `notebooks/00_setup_and_checks`. It creates `workspace.docs_rag` and its Volumes, then reports whether
   Hugging Face and the AI Gateway are reachable.
4. Upload the PDFs **and** `manifest.csv` to `/Volumes/workspace/docs_rag/raw_pdfs/dulux_datasheets/`
   (Catalog Explorer → Upload to this volume).
5. Run notebooks `01` → `05` in order.

### If Free Edition blocks a domain
Free Edition only allows outbound traffic to [a limited set of trusted domains](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).
Notebook 00 tells you which of these you need:

| Blocked | Fallback |
|---|---|
| `huggingface.co` (model download) | `python scripts/download_model.py` on your laptop, upload `models/bge-small-en-v1.5/` to `/Volumes/workspace/docs_rag/models/`, set `embedding.model_path` |
| `ai-gateway.vercel.sh` (LLM) | Run the answer/eval steps locally: `python scripts/run_local.py all && python scripts/run_local.py eval` (same code, Parquet instead of Delta). Or, if your workspace lists a chat model under **Serving**, point `llm.base_url` at `https://<workspace-host>/serving-endpoints` (also OpenAI-compatible) |

### Local run (no Spark)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export AI_GATEWAY_API_KEY=...           # never commit this
python scripts/run_local.py all         # bronze -> silver -> gold (Parquet in data/local_lake/)
python scripts/run_local.py ask "What is the recoat time for Wash&Wear 101 Low Sheen?"
python scripts/run_local.py eval
```

### Tests
```bash
pip install -r requirements-dev.txt
pytest -q tests      # offline: synthetic PDFs, fake embedder + fake LLM, local Spark for Bronze/Silver
```

## Demo corpus: DuluxGroup product data sheets

<!-- TODO: fill in the real counts after notebook 01 -->
- **[N] PDFs** from Dulux, Selleys and Cabot's: [n] Technical Data Sheets (TDS) and [n] Safety Data Sheets (SDS), [n] pages.
- Sources: publicly available documents from official DuluxGroup channels (e.g. the DuSpec+ specification site for
  TDS and the DuluxGroup SDS portal). `data/manifest.csv` records the source URL of every file.
- The PDFs themselves are **not committed** (third-party copyright). Run `scripts/download_pdfs.py` to fetch them.
- Why this corpus: data sheets are highly structured, full of tables, and contain many **near-identical products**
  (e.g. *Wash&Wear 101 Low Sheen* vs *Wash&Wear +Plus Kitchen & Bathroom*), which makes retrieval a real challenge.

## Example questions and answers

<!-- TODO: paste 3-4 real outputs from notebook 04 (answer + citations), including one "not in the documents" -->
```
Q: ...
A: ... [S1]
   [S1] <file>.pdf p.2 - Recoat Time
```

## Results

<!-- TODO: paste the table printed by notebook 05 -->
Evaluation on **[20] hand-checked questions** ([15] answerable, [5] deliberately unanswerable):

| run | k | metadata_filter | n_questions | retrieval_hit_rate | answer_accuracy | unanswerable_refusal_rate | citation_accuracy |
|---|---|---|---|---|---|---|---|
| k5_nofilter | 5 | False | 20 | – | – | – | – |
| k3_filter | 3 | True | 20 | – | – | – | – |
| k5_filter | 5 | True | 20 | – | – | – | – |

- **retrieval_hit_rate**: answerable questions where a chunk from the expected file (and page) is in the top-k.
- **answer_accuracy**: all questions judged correct; answerable ones by an LLM judge against the hand-written reference
  answer, unanswerable ones only if the system says "not in the documents".
- **citation_accuracy**: answerable questions whose citations include the expected file.

## Design decisions

**Section-aware chunking, not fixed-size windows.** Data sheets are already organised into sections (SDS: 16 numbered
sections; TDS: *Surface preparation*, *Drying time*, *Recoat*, ...). Chunks follow those boundaries and are only split
(with 40-word overlap) when a section exceeds 220 words. A fixed 500-character window would regularly separate
"Recoat: 2 hours" from the product and section it belongs to.

**A metadata header on every chunk's embedding text.** Each chunk is embedded as
`brand | product | doc type | section` followed by the text. Without this, the *Technical data* table of two sibling
products is almost the same text and gets almost the same vector.

**Metadata filtering when the question names a product.** Even with headers, similar products compete. If the question
names a known product (matched on normalised names, most specific first; "&" = "and", ® and ™ ignored), retrieval
searches only that product's chunks, falling back to the brand, then to an unfiltered search. The evaluation reports
results with and without the filter.

**Tables rendered as `key: value` lines.** Plain PDF text extraction reads tables row-by-row across columns and mixes
them with the text beside them. pdfplumber finds the table boxes; text outside them is extracted normally, and each
table row becomes `Property: value`.

**De-duplication at two levels, scoped carefully.** Documents: the same normalised text under two file names keeps one
(manifest entry first, then shortest name). Chunks: identical text is removed only **within the same product**.
Removing boilerplate across products would leave some products with no chunk for a product-filtered search to find.

**Grounded prompt with an explicit refusal string.** The model gets numbered sources with file/page/section, must cite
`[Sn]` after each fact, and must answer exactly "That information is not in the documents." when it can't. The fixed
string makes refusals measurable.

**FAISS by default, Databricks Vector Search optional.** A few thousand 384-dim vectors fit easily in exact
(brute-force) search. The Gold Delta table has Change Data Feed on, so `vector_backend: databricks_vs` builds a
Delta Sync index over the same table.

**Embeddings on the driver.** Loading the model once and encoding a few thousand chunks takes seconds; a Spark UDF
would load the model once per task. PDF parsing and chunking, which are per-document and independent, do run
distributed (`mapInPandas`, `applyInPandas`), with a driver fallback.

**One plain-Python package for everything.** `src/docs_rag` has no Spark dependency, so the same functions run inside
Spark UDFs, on the driver, on a laptop, and in unit tests.

## Problems found and how they were fixed

Found while testing the pipeline on synthetic data sheets (`tests/make_fixtures.py`):

| Problem | Fix |
|---|---|
| Footer `Company ... Page 1 of 2` leaked into chunks: repeated-line removal needed 3+ pages, and the changing page number made each footer line unique | compare lines with digits masked; apply from 2 pages; strip trailing `Page x of y` |
| Duplicate SDS: the pipeline kept `...-copy.pdf` and dropped the original (plain alphabetical order) | prefer the manifest entry, then the shortest file name |
| Product name taken from the file name (`Brightco Deckcoat`) instead of the title | read the SDS `Product name:` field, else the first title line, and strip the brand prefix |
| Re-joining hyphenated line breaks also merged ranges: `2-\n4 hours` → `24 hours` | only re-join letter-hyphen-letter |
| Title-only fragments (4–5 words) became their own chunks | merge fragments under `min_words` into a neighbouring chunk |
| "Wash&Wear Low Sheen" didn't match the product "Wash&Wear 101 Low Sheen" | match on 75% of distinctive tokens, most specific product first; ambiguous questions get no filter |
| `df.cache()` isn't supported on serverless compute | removed; the data is small enough to recompute |

<!-- TODO: add what you hit on the real PDFs, e.g. a table that still parsed badly, a product retrieved for its sibling,
     a scanned page, a question the model answered from outside knowledge. Each row = one interview story. -->
Found on the real corpus:

| Problem | Fix |
|---|---|
| | |

## Limitations
- No OCR: pages without a text layer are flagged in `silver_dq_checks` and skipped.
- The answer judge is an LLM; answerable questions were written and checked by hand, but judging is still automated.
- 20 questions is a small evaluation set; treat differences of one or two questions as noise.
- English-only; product matching relies on product names in the manifest or document title.
