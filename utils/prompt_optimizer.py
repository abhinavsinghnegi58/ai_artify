"""Utility helpers for rewriting user prompts with and without retrieval context."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from flask import current_app


def optimize_prompt(user_prompt: str) -> str:
    """
    Enhance the user prompt for better image generation results.

    Falls back to the original prompt if the optimization service
    fails or is unavailable.
    """
    prompt = user_prompt.strip()
    if not prompt:
        return user_prompt

    client = current_app.config.get("OPENAI_CLIENT")
    if client is None:
        current_app.logger.warning("OPENAI client not configured; skipping prompt optimization.")
        return user_prompt

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=_build_messages(prompt),
            max_tokens=120,
        )
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else user_prompt
    except Exception:  # pragma: no cover - defensive fallback
        current_app.logger.exception("Prompt optimization failed")
        return user_prompt


def optimize_with_context(user_prompt: str, retrieved_snippets: Iterable[str]) -> str:
    """
    Blend dense-retrieved snippets with the current user prompt.

    If no snippets are provided the function gracefully falls back
    to the standard prompt optimizer.
    """
    cleaned = [snippet.strip() for snippet in retrieved_snippets if snippet and snippet.strip()]
    if not cleaned:
        return optimize_prompt(user_prompt)

    client = current_app.config.get("OPENAI_CLIENT")
    if client is None:
        current_app.logger.warning("OPENAI client not configured; skipping contextual optimization.")
        return user_prompt

    snippet_block = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(cleaned[:5]))

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=_build_context_messages(user_prompt, snippet_block),
            max_tokens=180,
        )
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else user_prompt
    except Exception:  # pragma: no cover - defensive fallback
        current_app.logger.exception("Contextual prompt optimization failed")
        return user_prompt


def _build_messages(prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are an expert creative prompt engineer for AI image generation.",
        },
        {
            "role": "user",
            "content": f"Improve this DALL-E prompt for clarity, style, and creativity:\n{prompt}",
        },
    ]


def _build_context_messages(prompt: str, snippets: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert creative director who fuses user prompts with prior inspiration "
                "to generate one vivid, production-ready prompt for AI image generation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original prompt:\n"
                f"{prompt}\n\n"
                "Reference inspirations retrieved from the user's knowledge base:\n"
                f"{snippets}\n\n"
                "Produce a single refined prompt (no bullet list) that blends the best attributes "
                "from the references while preserving the user's intent."
            ),
        },
    ]
