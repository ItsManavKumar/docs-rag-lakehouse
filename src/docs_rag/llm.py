"""OpenAI-compatible chat client (Vercel AI Gateway by default).

The API key is read from Databricks secrets or an environment variable.
It is never written to config, notebooks or logs.
"""
from __future__ import annotations

import os
import time


def get_api_key(cfg: dict, dbutils=None) -> str:
    llm = cfg["llm"]
    if dbutils is not None:
        try:
            return dbutils.secrets.get(scope=llm["secret_scope"], key=llm["secret_key"])
        except Exception:
            pass
    key = os.environ.get(llm["api_key_env"])
    if not key:
        raise RuntimeError(
            f"No API key found. Set Databricks secret {llm['secret_scope']}/{llm['secret_key']} "
            f"or environment variable {llm['api_key_env']}.")
    return key


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 700):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], model: str | None = None, retries: int = 3) -> str:
        last = None
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=model or self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens)
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limits / transient network errors
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")


class FakeChatClient:
    """Offline stand-in for tests: answers with the first retrieved source, or
    'not in the documents' if nothing was retrieved. TEST USE ONLY."""

    model = "fake-test-llm"

    def chat(self, messages: list[dict], model: str | None = None, retries: int = 1) -> str:
        user = messages[-1]["content"]
        if "GRADE" in messages[0]["content"]:
            return '{"correct": true, "reason": "fake judge"}'
        if "[S1]" not in user:
            return "That information is not in the documents."
        block = user.split("[S1]", 1)[1].split("\n[S2]", 1)[0]
        body = " ".join(block.split("\n")[1:])[:160].strip()
        return f"{body} [S1]"


def build_client(cfg: dict, dbutils=None, test_mode: bool = False):
    if test_mode:
        return FakeChatClient()
    llm = cfg["llm"]
    return ChatClient(llm["base_url"], get_api_key(cfg, dbutils), llm["model"],
                      llm.get("temperature", 0), llm.get("max_tokens", 700))
