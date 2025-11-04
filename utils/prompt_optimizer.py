from __future__ import annotations

from typing import Any, Dict, List

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
