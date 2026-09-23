# docs-rag-lakehouse

A retrieval-augmented generation (RAG) pipeline on **Databricks** that answers questions over a folder of PDFs
and cites the document and page for every fact. It's built as a **medallion Lakehouse** (Bronze → Silver → Gold Delta
tables) with **PySpark**, and evaluated with a question set whose answers were checked by hand.

It works with any set of documents: `config/config.yaml` points at a folder of PDFs, so the corpus can be swapped
without code changes.

> Independent portfolio project built over **publicly available** product data sheets. Not affiliated with,
> endorsed by, or built for DuluxGroup.

**Measured on 29 hand-checked questions:** retrieval accuracy 55% → 100% with metadata filtering,
97% answer accuracy, and 100% correct refusals on the 7 questions the documents can't answer.
[Full results](#results).

---

## Architecture

```mermaid
flowchart LR
    A[PDFs in a UC Volume<br/>+ manifest.csv] -->|binaryFile + mapInPandas| B[(Bronze<br/>bronze_pages<br/>1 row per page)]
    B -->|clean · dedupe · section-aware chunking<br/>applyInPandas + window fns| C[(Silver<br/>silver_chunks<br/>tagged chunks)]
    C -->|bge-small-en-v1.5| D[(Gold<br/>gold_chunk_embeddings<br/>Delta + CDF)]
    D --> E[Hybrid retrieval<br/>FAISS vectors + BM25 keywords<br/>merged with RRF]
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
                          download_model.py, draft_eval_questions.py, inspect_sections.py
eval/questions.jsonl      evaluation questions (answers checked against the PDFs)
tests/                    unit tests (chunking, filters, hybrid retrieval, PDF layouts) + a
                          local-Spark test of Bronze/Silver on synthetic PDFs
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

- **34 public documents** from Dulux, Cabot's and Selleys: 29 Technical Data Sheets and 5 Safety
  Data Sheets, 138 pages, producing 532 chunks.
- Sources: publicly available documents from official DuluxGroup channels (the DuSpec+ specification site for TDS, the
  DuluxGroup SDS portal, and the Selleys and Cabot's product sites). `data/manifest.csv` records the source URL of
  every file.
- The PDFs themselves are **not committed** (third-party copyright). Run `scripts/download_pdfs.py` to fetch them.
- Why this corpus: data sheets are highly structured, full of tables, and contain many **near-identical products**
  (Aquanamel Gloss / Semi Gloss / Low Gloss, four Weathershield finishes, four Cabothane variants), which makes
  retrieval a real challenge.

## Example questions and answers

Real output from `notebooks/04_rag_query`, running on Databricks:

```
Q: What is the touch dry time for Aquanamel Gloss?
A: Touch dry time for Dulux Aquanamel Gloss is 30 Minutes [S4].
   (metadata filter: {'product': ['Aquanamel Gloss']})
   [S4] dulux-aquanamel-gloss-tds.pdf p.2 - Typical Properties

Q: What is the touch dry time for Aquanamel Semi Gloss?
A: The touch dry time for Dulux Aquanamel Semi Gloss is 30 Minutes [S4], though the full film
   requires 7 days to fully cure before cleaning [S1].
   (metadata filter: {'product': ['Aquanamel Semi Gloss']})
   [S1] dulux-aquanamel-semi-gloss-tds.pdf pp.1-2 - Precautions and Limitations
   [S4] dulux-aquanamel-semi-gloss-tds.pdf p.2 - Typical Properties

Q: How much does a 4 litre tin of Weathershield Low Sheen cost?
A: That information is not in the documents. The sources cover the product's weight
   (5.5 Kg for 4 Litre) [S2] but not pricing.
   [S2] dulux-weathershield-low-sheen-tds.pdf p.4 - Transport and Storage
```

The first two are sibling products whose data sheets are nearly identical; each answer is drawn from
its own document. The third is a question the corpus cannot answer.

## Results

29 questions, each written by hand with its answer read from the source PDF: 22 answerable,
7 deliberately unanswerable (a price, a retailer, a real Dulux product outside the corpus, a
technology none of the documents mention, and colour advice).

| run | k | metadata_filter | n_questions | retrieval_hit_rate | answer_accuracy | unanswerable_refusal_rate | citation_accuracy |
|---|---|---|---|---|---|---|---|
| k5_nofilter | 5 | False | 29 | 55% | 52% | 100% | 86% |
| k3_filter | 3 | True | 29 | 96% | 93% | 100% | 100% |
| k5_filter | 5 | True | 29 | 100% | 97% | 100% | 100% |

- **retrieval_hit_rate**: answerable questions where a chunk from the expected file and page is in the top-k.
- **answer_accuracy**: all questions judged correct; answerable ones by an LLM judge against the hand-written reference
  answer, unanswerable ones only if the system says "not in the documents".
- **citation_accuracy**: answerable questions whose citations include the expected file.

**Reading this:** metadata filtering is what moves the numbers — retrieval hit rate 55% → 100%.
Without it, a question about one product routinely retrieves a sibling's data sheet.
Refusals were perfect in every configuration: the system never invented a price, a retailer or a
warranty, which is the behaviour that matters most for product safety information.

**The one failure at k=5 with filtering** is question q04, the recoat time for 1 Step Prep. The source
table has Min / Max / Recommended columns, so the row reads `Recoat Time 2 hours | Indefinite | 2 hours`.
Extraction flattens the table to a single line, the column headers are lost, and the model attributed
"indefinite" to roller application rather than to the maximum column. The values survive; their meaning
doesn't. Carrying column headers through extraction is the fix.

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

**Hybrid retrieval: vector plus keyword.** Vector search alone missed a question whose answer was in
the corpus: the recoat time for 1 Step Prep sits inside a long passage about film thickness and spread
rates, titled "Clean Up", and ranked 9th for "What is the recoat time...?". The literal phrase
"Recoat Time" was right there. So retrieval now runs BM25 keyword search alongside vector search and
merges the two rankings with Reciprocal Rank Fusion. Two details mattered: the product name is stripped
from the keyword query (every chunk of that product contains it, and BM25 favours short documents), and
stopwords are removed (otherwise "what is the ... for" matches short chunks on common words). Each
result records whether it was found by `vector`, `keyword` or `both`.

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

Found on the real corpus (34 public data sheets):

| Problem | Fix |
|---|---|
| **67% of chunks (315 of 467) had no recognised section.** DuSpec+ data sheets (Dulux, Cabot's) use Title Case headings with no colon (`Description and Image`) and inline headings (`Uses: Use Aquanamel on...`); the detector only knew ALL-CAPS and `Heading:` lines | recognise title-case headings that contain a section keyword, and split inline `Heading: sentence` lines (only when the rest is a real sentence, so `Recoat: 2 hours` table rows stay content). After: **15% (80 of 533)** |
| **Two-column Selleys sheets were read straight across**, merging unrelated sentences (`Approvals & Standards with skin and eyes.`) | detect a vertical gutter and read left column then right, band by band, so full-width titles, tables and footers stay in order |
| **Tables were appended at the end of each page**, so a drying-time table could land under the last heading on the page | place each table back at its vertical position among the text lines |
| 3 of 37 source links failed: 2 returned a CAPTCHA page instead of a PDF, 1 was a 404 | download script checks every file starts with `%PDF`; failed documents were removed from the manifest so it lists exactly the corpus |
| Hundreds of `Could not get FontBBox` warnings from malformed fonts in some PDFs | harmless; silenced pdfminer's logger |
| Even after the layout fixes, several genuine headings still fell into "General": SDS sub-headings (`Small Spills`, `Dangerous Good Classification`, `Chemical Entity Cas No Proportion`), TDS `Introduction`/`Product Information` blocks, and Selleys `Standards & Certificates` — either unmapped in `CANONICAL` or (for `Standards & Certificates`) not yet in `TDS_HEADINGS` | added `Standards & Certificates` as a recognised heading, and new `CANONICAL` keyword rules (`spill`→Spills, `dangerous good`→Transport, `cas no`/`proportion`→Composition, `introduction`→Product description, `approv`/`standard`/`certif`→Approvals, `product information`→Technical data, `maintenance`→Maintenance) |
| DuSpec+ page frames with panel dividers were read as a one-column table covering the page, squashing each panel (heading + property table) into one line | reject "tables" with fewer than 2 rows or 2 columns of content, or covering >70% of the page |
| An SDS title drawn with wide letter spacing was read as two columns and split mid-word (`Weathershield` → `Weat` + `hershield`) | a column gutter must be at least 14 points wide; letter spacing inside a word is narrower |
| A rare exact phrase buried in a long noisy chunk ranked 9th under vector search (`Recoat Time` inside the 1 Step Prep "Clean Up" passage) | hybrid retrieval: BM25 keyword search merged with vector search by Reciprocal Rank Fusion |

## Limitations
- No OCR: pages without a text layer are flagged in `silver_dq_checks` and skipped.
- The answer judge is an LLM; answerable questions were written and checked by hand, but judging is still automated.
- At k=5 with metadata filtering the question set is nearly saturated (one failure in 29), so it can no
  longer discriminate between good and better configurations. Harder questions are the next step:
  questions that name no product, ambiguous product references, and answers that span a TDS and an SDS.
- Multi-column tables (Min / Max / Recommended) are flattened during extraction, which preserves the
  values but loses which column each belongs to.
- The evaluation runs sequentially, roughly 150 model calls in about 9 minutes. It would parallelise
  easily; it just hasn't needed to.
- English-only; product matching relies on product names in the manifest or document title.
