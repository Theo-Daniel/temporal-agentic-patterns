from unittest.mock import AsyncMock, MagicMock, patch

from expense_agent.ollama_client import DEFAULT_TEMPERATURE, call_ollama_chat, call_ollama_chat_streaming


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"message": {"content": content}}
    response.raise_for_status.return_value = None
    return response


async def test_call_ollama_chat_sends_format_json_and_default_temperature():
    with patch(
        "expense_agent.ollama_client.requests.post", return_value=_mock_response("hello")
    ) as mock_post:
        result = await call_ollama_chat([{"role": "user", "content": "hi"}], model="test-model")

    assert result == "hello"
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == DEFAULT_TEMPERATURE


async def test_call_ollama_chat_allows_temperature_override():
    with patch(
        "expense_agent.ollama_client.requests.post", return_value=_mock_response("hello")
    ) as mock_post:
        await call_ollama_chat([{"role": "user", "content": "hi"}], temperature=0.5)

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["options"]["temperature"] == 0.5


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        pass


class _FakeStreamContextManager:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


async def test_call_ollama_chat_streaming_invokes_on_delta_in_order_and_returns_full_text():
    lines = [
        '{"message": {"content": "Hel"}, "done": false}',
        '{"message": {"content": "lo"}, "done": false}',
        '{"message": {"content": ""}, "done": true}',
    ]
    fake_response = _FakeStreamResponse(lines)
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_FakeStreamContextManager(fake_response))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    deltas = []

    async def on_delta(delta):
        deltas.append(delta)

    with patch("expense_agent.ollama_client.httpx.AsyncClient", return_value=fake_client):
        result = await call_ollama_chat_streaming(
            [{"role": "user", "content": "hi"}], on_delta, model="test-model"
        )

    assert deltas == ["Hel", "lo"]
    assert result == "Hello"


async def test_call_ollama_chat_streaming_sends_format_json_and_temperature():
    lines = ['{"message": {"content": "x"}, "done": true}']
    fake_response = _FakeStreamResponse(lines)
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_FakeStreamContextManager(fake_response))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    async def on_delta(delta):
        pass

    with patch("expense_agent.ollama_client.httpx.AsyncClient", return_value=fake_client):
        await call_ollama_chat_streaming(
            [{"role": "user", "content": "hi"}], on_delta, temperature=0.3
        )

    _, kwargs = fake_client.stream.call_args
    payload = kwargs["json"]
    assert payload["stream"] is True
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0.3
