"""docs_rag: a small, document-agnostic RAG pipeline used by the Databricks notebooks.

Everything in here is plain Python (no Spark imports), so the same functions run
inside Spark UDFs, on the driver, or on a laptop via scripts/run_local.py.
"""

__version__ = "0.1.0"
