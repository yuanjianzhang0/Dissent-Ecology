import io
import json

import pytest

from dissent_ecology.backend import ChatBackend
from dissent_ecology.protocol import Problem


def test_endpoint_request_preserves_independent_audit_and_usage(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "REPLACE #### 4"}}],
                                     "usage": {"completion_tokens": 8, "prompt_tokens": 60}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    problem = Problem("p", "What is two plus two?", "It is three.", "3", 5,
                      {"arithmetic": {"expression": "2+2", "private_fact": "not for the critic"}})
    backend = ChatBackend(api_key="test-key", extra_body={"seed": 42})
    result = backend.audit(problem, "arithmetic", 30)
    body = json.loads(requests[0].data)
    assert body["temperature"] == 0 and body["max_tokens"] == 30 and body["seed"] == 42
    assert body["n"] == 1 and body["stream"] is False
    assert "private_fact" not in requests[0].data.decode()
    assert result.tokens == 8 and result.prompt_tokens == 60
    assert result.action.replacement == "4"
    assert requests[0].get_header("Authorization") == "Bearer test-key"
    with pytest.raises(ValueError, match="exceeded"):
        backend.audit(problem, "arithmetic", 4)


def test_required_decoding_fields_cannot_be_overridden():
    with pytest.raises(ValueError):
        ChatBackend(extra_body={"temperature": 1})
    with pytest.raises(ValueError):
        ChatBackend(base_url="file:///tmp")
