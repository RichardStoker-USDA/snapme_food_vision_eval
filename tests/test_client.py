"""Request settings for an endpoint the benchmark did not start: key file, extra request fields."""
import asyncio

from snapme_eval.client import Chat
from snapme_eval.runner import sampling_for


def test_key_file_sets_the_header(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("abc123\n")
    monkeypatch.setenv("SNAPME_API_KEY_FILE", str(key))
    chat = Chat("http://127.0.0.1:1/v1", "m", {})
    assert chat.http.headers["Authorization"] == "Bearer abc123"
    asyncio.run(chat.close())


def test_no_key_file_keeps_the_placeholder(monkeypatch):
    monkeypatch.delenv("SNAPME_API_KEY_FILE", raising=False)
    chat = Chat("http://127.0.0.1:1/v1", "m", {})
    assert chat.http.headers["Authorization"] == "Bearer none"
    asyncio.run(chat.close())


def test_request_table_joins_the_body():
    m = {"reasoning": "optional", "sampling": {"temperature": 0.0}, "request": {"priority": 10}}
    sampling, extra = sampling_for(m)
    assert sampling == {"temperature": 0.0}
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}, "priority": 10}


def test_template_switches_override_the_default():
    m = {"reasoning": "optional", "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    assert sampling_for(m)[1] == {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
