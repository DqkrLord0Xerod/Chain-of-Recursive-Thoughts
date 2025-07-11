import json
import requests
import aiohttp

from typing import Optional

from exceptions import APIError, RateLimitError, TokenLimitError
from config import settings

BASE_URL = settings.api_base_url
EMBED_URL = settings.embed_url


def _handle_response(response):
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded")
    if response.status_code >= 400:
        try:
            data = response.json()
        except Exception:
            data = {}
        msg = data.get("error") if isinstance(data, dict) else response.text
        if "token" in str(msg).lower():
            raise TokenLimitError(str(msg))
        raise APIError(str(msg))


def sync_chat_completion(headers, messages, model, temperature=0.7, stream=True):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "reasoning": {"max_tokens": 10386},
    }
    response = requests.post(
        BASE_URL, headers=headers, json=payload, stream=stream
    )
    _handle_response(response)
    if stream:
        full = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full += content
        return full
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def async_chat_completion(
    headers,
    messages,
    model,
    temperature=0.7,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "reasoning": {"max_tokens": 10386},
    }

    async def _post(sess: aiohttp.ClientSession) -> str:
        async with sess.post(BASE_URL, headers=headers, json=payload) as resp:
            if resp.status == 429:
                raise RateLimitError("Rate limit exceeded")
            if resp.status >= 400:
                try:
                    data = await resp.json()
                except Exception:
                    data = {}
                msg = data.get("error") if isinstance(data, dict) else await resp.text()
                if "token" in str(msg).lower():
                    raise TokenLimitError(str(msg))
                raise APIError(str(msg))
            data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

    if session is None:
        async with aiohttp.ClientSession() as sess:
            return await _post(sess)
    return await _post(session)


def get_embeddings(headers, texts):
    payload = {"model": "openai/text-embedding-ada-002", "input": texts}
    resp = requests.post(EMBED_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [d["embedding"] for d in data["data"]]
