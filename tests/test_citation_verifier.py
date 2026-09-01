
from rag.citation_verifier import verify_citations


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(
        "rag.citation_verifier._make_llm_client", lambda: fake_async_client
    )


class TestVerifyCitationsEmptyAnswer:
    async def test_empty_answer_fails_open_without_calling_llm(self, monkeypatch, fake_async_client):
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("", ["frag"])

        assert result == {"invalid_citations": [], "verdict": "clean"}
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_whitespace_only_answer_fails_open_without_calling_llm(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("   \n  ", ["frag"])

        assert result == {"invalid_citations": [], "verdict": "clean"}
        fake_async_client.chat.completions.create.assert_not_called()


class TestVerifyCitationsValidResponses:
    async def test_clean_verdict_no_invalid_citations(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"invalid_citations": [], "verdict": "clean"}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["fragment text"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_minor_verdict_with_invalid_citation(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = (
            '{"invalid_citations": [{"index": 2, "claim": "some claim", '
            '"reason": "not found"}], "verdict": "minor"}'
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1] [2].", ["a", "b"])

        assert result == {
            "invalid_citations": [{"index": 2, "claim": "some claim", "reason": "not found"}],
            "verdict": "minor",
        }

    async def test_unsupported_verdict(self, monkeypatch, fake_async_client, fake_llm_response):
        content = (
            '{"invalid_citations": [{"index": 1, "claim": "x", "reason": "y"}, '
            '{"index": 2, "claim": "z", "reason": "w"}], "verdict": "unsupported"}'
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1] [2].", ["a", "b"])

        assert result["verdict"] == "unsupported"
        assert len(result["invalid_citations"]) == 2

    async def test_response_wrapped_in_json_fence(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '```json\n{"invalid_citations": [], "verdict": "clean"}\n```'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}


class TestVerifyCitationsRequestFormat:
    async def test_uses_json_schema_when_supported(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"invalid_citations": [], "verdict": "clean"}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)
        monkeypatch.setattr("rag.citation_verifier.supports_json_schema", lambda: True)
        monkeypatch.setattr("rag.citation_verifier.supports_json_object", lambda: False)

        await verify_citations("Answer [1].", ["a"])

        _, kwargs = fake_async_client.chat.completions.create.call_args
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "citation_audit"

    async def test_uses_json_object_when_schema_unsupported(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"invalid_citations": [], "verdict": "clean"}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)
        monkeypatch.setattr("rag.citation_verifier.supports_json_schema", lambda: False)
        monkeypatch.setattr("rag.citation_verifier.supports_json_object", lambda: True)

        await verify_citations("Answer [1].", ["a"])

        _, kwargs = fake_async_client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}

    async def test_no_response_format_when_neither_supported(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"invalid_citations": [], "verdict": "clean"}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)
        monkeypatch.setattr("rag.citation_verifier.supports_json_schema", lambda: False)
        monkeypatch.setattr("rag.citation_verifier.supports_json_object", lambda: False)

        await verify_citations("Answer [1].", ["a"])

        _, kwargs = fake_async_client.chat.completions.create.call_args
        assert "response_format" not in kwargs


class TestVerifyCitationsFailOpen:
    async def test_empty_llm_content_fails_open(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_content_without_json_object_fails_open(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("no json here")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_invalid_verdict_defaults_to_clean(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"invalid_citations": [], "verdict": "weird"}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_client_exception_fails_open(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_make_client_raising_fails_open(self, monkeypatch):
        def _raise():
            raise RuntimeError("no client")

        monkeypatch.setattr("rag.citation_verifier._make_llm_client", _raise)

        result = await verify_citations("Answer [1].", ["a"])

        assert result == {"invalid_citations": [], "verdict": "clean"}

    async def test_malformed_invalid_citation_item_is_skipped(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = (
            '{"invalid_citations": [{"claim": "missing index field"}, '
            '{"index": 3, "claim": "ok", "reason": "r"}], "verdict": "minor"}'
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_citations("Answer [3].", ["a", "b", "c"])

        assert result["verdict"] == "minor"
        assert result["invalid_citations"] == [{"index": 3, "claim": "ok", "reason": "r"}]
