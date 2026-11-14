import asyncio
import json

import httpx
import requests

from expense_agent.config import OLLAMA_CHAT_URL, OLLAMA_MODEL

DEFAULT_TEMPERATURE = 0.1


def _post_chat(messages: list, model: str, temperature: float) -> str:
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


async def call_ollama_chat(
    messages: list, model: str = OLLAMA_MODEL, temperature: float = DEFAULT_TEMPERATURE
) -> str:
    return await asyncio.to_thread(_post_chat, messages, model, temperature)


async def call_ollama_chat_streaming(
    messages: list,
    on_delta,
    model: str = OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    full_text = ""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "format": "json",
                "options": {"temperature": temperature},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    full_text += delta
                    await on_delta(delta)
    return full_text
