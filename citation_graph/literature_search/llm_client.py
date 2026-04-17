"""
LLM client helpers for the literature search pipeline.
Uses OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY from env.
Returns a generate_fn(system_message, user_prompt, pdf_paths=None) -> response_text.
When pdf_paths is provided and provider is OpenRouter, sends whole PDFs to the API (multimodal).
"""

import base64
import os
from pathlib import Path
from typing import Callable, List, Optional


def get_llm_generate_fn(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Callable[..., str]]:
    """
    Return a sync function (system_message, user_prompt) -> response_text, or None if no API key.
    provider can override: "openai" | "anthropic" | "openrouter". If unset, prefer by env key presence.
    """
    use_openrouter = (provider == "openrouter") or (provider is None and os.environ.get("OPENROUTER_API_KEY"))
    use_openai = (provider == "openai") or (provider is None and os.environ.get("OPENAI_API_KEY"))
    use_anthropic = (provider == "anthropic") or (provider is None and os.environ.get("ANTHROPIC_API_KEY"))

    if use_openrouter and os.environ.get("OPENROUTER_API_KEY"):
        return _make_openrouter_fn(model or "anthropic/claude-sonnet-4.6")
    if use_openai and os.environ.get("OPENAI_API_KEY"):
        return _make_openai_fn(model or "gpt-5.4")
    if use_anthropic and os.environ.get("ANTHROPIC_API_KEY"):
        return _make_anthropic_fn(model or "claude-sonnet-4-6")
    return None


def _make_openai_fn(model: str) -> Callable[..., str]:
    def fn(
        system_message: str,
        user_prompt: str,
        pdf_paths: Optional[List[str]] = None,
    ) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            if r.choices and r.choices[0].message.content:
                return r.choices[0].message.content
        except Exception as e:
            print(f"[literature_search] OpenAI error: {e}")
        return ""

    return fn


def _make_anthropic_fn(model: str) -> Callable[..., str]:
    def fn(
        system_message: str,
        user_prompt: str,
        pdf_paths: Optional[List[str]] = None,
    ) -> str:
        try:
            from anthropic import Anthropic
            client = Anthropic()
            r = client.messages.create(
                model=model,
                max_tokens=4000,
                system=system_message,
                messages=[{"role": "user", "content": user_prompt}],
            )
            if r.content and r.content[0].type == "text":
                return r.content[0].text
        except Exception as e:
            print(f"[literature_search] Anthropic error: {e}")
        return ""

    return fn


def _make_openrouter_fn(model: str) -> Callable[..., str]:
    """OpenRouter: text-only or PDFs via type=file (e.g. Sonnet 4.6). pdf_paths=list of paths to send whole PDFs."""
    def fn(
        system_message: str,
        user_prompt: str,
        pdf_paths: Optional[List[str]] = None,
    ) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[literature_search] OPENROUTER_API_KEY not set")
            return ""
        pdf_paths = pdf_paths or []
        # Build user content: text + optional file parts (whole PDFs)
        user_content: List[dict] = [{"type": "text", "text": user_prompt}]
        for i, path in enumerate(pdf_paths):
            p = Path(path)
            if not p.exists():
                continue
            try:
                raw = p.read_bytes()
                b64 = base64.b64encode(raw).decode("utf-8")
                data_url = f"data:application/pdf;base64,{b64}"
                user_content.append({
                    "type": "file",
                    "file": {
                        "filename": p.name or f"paper_{i+1}.pdf",
                        "file_data": data_url,
                    },
                })
            except Exception as e:
                print(f"[literature_search] OpenRouter: skip PDF {path}: {e}")
        try:
            import requests
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,
                "max_tokens": 4000,
            }
            # Optional: use pdf-text parser when model doesn't support native PDF
            if user_content and any(c.get("type") == "file" for c in user_content):
                payload["plugins"] = [
                    {"id": "file-parser", "pdf": {"engine": "pdf-text"}},
                ]
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content")
                if content:
                    return content.strip()
        except Exception as e:
            print(f"[literature_search] OpenRouter error: {e}")
        return ""

    return fn
