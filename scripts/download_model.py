"""Save the embedding model to a local folder (run on your laptop).

Only needed if Databricks Free Edition can't reach huggingface.co (notebook 00 tells you).

  python scripts/download_model.py            # -> models/bge-small-en-v1.5/

Then upload the folder to /Volumes/workspace/docs_rag/models/bge-small-en-v1.5
(Catalog Explorer > Upload, or `databricks fs cp -r models/bge-small-en-v1.5 dbfs:/Volumes/workspace/docs_rag/models/bge-small-en-v1.5`)
and set `embedding.model_path` in config/config.yaml to that path.
"""
import sys

from sentence_transformers import SentenceTransformer

name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
out = f"models/{name.split('/')[-1]}"
SentenceTransformer(name).save(out)
print("saved to", out)
