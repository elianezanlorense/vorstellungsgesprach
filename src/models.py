
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb
from google import genai


_SKIP_SUBSTRINGS = [
    "embedding",
    "imagen",
    "veo",
    "tts",
    "audio",
    "robotics"
]


def list_available_chat_models(client: genai.Client) -> list[str]:
    """All models available in the GenAI API that can generate text."""
    available_models: list[str] = []

    for model in client.models.list():
        model_name = model.name

        if not model_name:
            continue

        if any(
            substring in model_name.casefold()
            for substring in _SKIP_SUBSTRINGS
        ):
            continue

        try:
            chat = client.chats.create(model=model_name)
            response = chat.send_message("Reply only with OK.")

            if response.text:
                available_models.append(model_name)

        except Exception:
            continue

    return available_models