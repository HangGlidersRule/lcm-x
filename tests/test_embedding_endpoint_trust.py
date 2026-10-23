"""Opt-in self-hosted embedding endpoint trust (#1).

The setting relaxes raw-text consent and enables automatic chunk embedding
for a private-network endpoint. It does not change the provider-bound privacy
transform or the privacy revision.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_maintenance as maintenance_mod
import hermes_lcm.ingest_protection as ingest_protection_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.ingest_protection import (
    embedding_endpoint_trust_status,
    embedding_privacy_revision,
    protect_embedding_text,
    sensitive_pattern_status,
)
from hermes_lcm.vector_store import VectorStore

_PREEXISTING_STATUS_KEYS = {
    "sensitive_patterns_enabled",
    "enabled",
    "sensitive_patterns",
    "patterns",
    "active_patterns",
    "unknown_patterns",
    "source",
    "embedding_privacy_setting",
    "embedding_privacy_enabled",
    "embedding_privacy_provider_requires_privacy",
    "embedding_privacy_revision",
    "placeholder_format",
    "lossless_recovery",
}

_CONSENT_ERROR = (
    "the chunk corpus derives provider input from RAW, VERBATIM message text — including "
    "tool-result and error/traceback content that the summary corpus "
    "never exposes — to the 'openai-compatible' cloud embedding provider. "
    "The v0.23.1 provider-input privacy transform removes configured "
    "detector matches but cannot classify every possible sensitive fact. "
    "Re-run with `--confirm-raw-text` to acknowledge the residual content "
    "boundary, or switch to a local provider (fastembed/ollama)."
)


class FakeProvider:
    provider_id = "openai-compatible"
    model_id = "work-bifrost/vllm/qwen3-embedding-8b"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        current = list(texts)
        self.calls.append(current)
        return [[float(index + 1), 1.0] for index, _text in enumerate(current)]


def _config(
    *,
    provider: str = "openai-compatible",
    base_url: str = "http://10.2.10.10:4000/v1",
    trust: str = "self-hosted",
    ollama_base_url: str | None = None,
):
    kwargs = {
        "embeddings_enabled": True,
        "embedding_provider": provider,
        "embedding_model": "work-bifrost/vllm/qwen3-embedding-8b",
        "embedding_base_url": base_url,
        "embedding_endpoint_trust": trust,
    }
    if ollama_base_url is not None:
        kwargs["ollama_base_url"] = ollama_base_url
    return LCMConfig(**kwargs)


def _reset_trust_warning() -> None:
    ingest_protection_mod._ENDPOINT_TRUST_REFUSAL_WARNED = False


@pytest.mark.parametrize(
    "url,reason",
    [
        ("http://10.2.10.10:4000/v1", "10.2.10.10 is private"),
        ("http://192.168.1.5/v1", "192.168.1.5 is private"),
        ("http://172.16.0.9:8080", "172.16.0.9 is private"),
        ("http://100.105.165.113:4000/v1", "100.105.165.113 is in 100.64.0.0/10"),
        ("http://127.0.0.1:11434", "127.0.0.1 is loopback"),
        ("http://localhost:8000/v1", "localhost"),
    ],
)
def test_helper_honors_literal_private_openai_compatible_endpoints(url, reason):
    honored, detail = embedding_endpoint_trust_status(
        _config(base_url=url, trust="self-hosted")
    )
    assert (honored, detail) == (True, reason)


@pytest.mark.parametrize("provider", ["openai", "siliconflow", "OpenAI-Compatible"])
def test_helper_honors_openai_family_aliases_on_a_private_endpoint(provider):
    honored, detail = embedding_endpoint_trust_status(
        _config(provider=provider, trust=" Self-Hosted ")
    )
    assert (honored, detail) == (True, "10.2.10.10 is private")


def test_helper_honors_ollama_with_a_private_base_url():
    # embedding_base_url stays on a public host so the ollama URL is the one judged.
    honored, detail = embedding_endpoint_trust_status(
        _config(
            provider="ollama",
            trust="self-hosted",
            base_url="https://api.siliconflow.cn/v1",
            ollama_base_url="http://192.168.1.9:11434",
        )
    )
    assert (honored, detail) == (True, "192.168.1.9 is private")


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (
            {"provider": "siliconflow", "base_url": "https://api.siliconflow.cn/v1"},
            "hostname api.siliconflow.cn is not a literal address",
        ),
        (
            {"base_url": "https://8.8.8.8/v1"},
            "8.8.8.8 is a public address",
        ),
        (
            {"base_url": "http://my-nas.local:4000"},
            "hostname my-nas.local is not a literal address",
        ),
        ({"base_url": ""}, "embedding endpoint URL is empty"),
        ({"base_url": "http://["}, "embedding endpoint URL is unparsable"),
        (
            {"provider": "voyage", "base_url": "http://10.2.10.10:4000/v1"},
            "provider voyage is not eligible",
        ),
        (
            {"provider": "fastembed", "base_url": "http://127.0.0.1:8000/v1"},
            "provider fastembed is already local",
        ),
        (
            {"trust": "auto", "base_url": "http://10.2.10.10:4000/v1"},
            "setting is auto",
        ),
    ],
)
def test_helper_refuses_untrusted_endpoints(kwargs, reason):
    _reset_trust_warning()
    honored, detail = embedding_endpoint_trust_status(_config(**kwargs))
    assert honored is False
    assert detail == reason


def test_unrecognized_setting_is_auto_and_warns_once(caplog):
    _reset_trust_warning()
    config = _config(trust="garbage", base_url="http://10.2.10.10:4000/v1")
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.ingest_protection"):
        first = embedding_endpoint_trust_status(config)
        second = embedding_endpoint_trust_status(config)
    assert first == (
        False,
        "unrecognized setting 'garbage'; treated as auto",
    )
    assert second == first
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and "garbage" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_refused_self_hosted_warns_once_with_the_reason(caplog):
    _reset_trust_warning()
    config = _config(trust="self-hosted", base_url="https://8.8.8.8/v1")
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.ingest_protection"):
        first = embedding_endpoint_trust_status(config)
        embedding_endpoint_trust_status(config)
    assert first == (False, "8.8.8.8 is a public address")
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "8.8.8.8 is a public address" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_auto_private_endpoint_is_not_honored_and_does_not_warn(caplog):
    _reset_trust_warning()
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.ingest_protection"):
        honored, detail = embedding_endpoint_trust_status(
            _config(trust="auto", base_url="http://10.2.10.10:4000/v1")
        )
    assert (honored, detail) == (False, "setting is auto")
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_resolver_underscore_alias_is_not_accepted():
    # resolve_provider does not accept openai_compatible; do not invent the alias.
    _reset_trust_warning()
    honored, detail = embedding_endpoint_trust_status(
        _config(provider="openai_compatible", trust="self-hosted")
    )
    assert honored is False
    assert detail == "unknown provider openai_compatible"


def _seed_chunk_profile(tmp_path, config):
    db_path = tmp_path / "endpoint-trust.db"
    config.database_path = str(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(1, 'sess-a', 'history', 'user', ?, 1.0)",
        ("u" * 60,),
    )
    conn.commit()
    conn.close()
    store = VectorStore(db_path, config=config)
    try:
        store.register_profile(
            config.embedding_model,
            "openai-compatible",
            2,
            revision=embedding_privacy_revision(config),
            task="chunk",
        )
    finally:
        store.close()
    return db_path


def test_honored_chunk_apply_is_not_refused_for_consent(monkeypatch, tmp_path):
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))
    import hermes_lcm.chunking as chunking

    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(str(text)))
    config = _config(trust="self-hosted")
    db_path = _seed_chunk_profile(tmp_path, config)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

    result = command_mod._chunk_backfill_run(
        config,
        db_path,
        apply=True,
        limit=10,
        retry_uncertain=False,
        policy="",
        confirm_raw_text=False,
    )

    assert not isinstance(result, str)
    assert "RAW, VERBATIM" not in str(result)
    assert result["embedded"] >= 1
    assert provider.calls


def test_auto_chunk_apply_is_refused_with_the_existing_consent_message(monkeypatch, tmp_path):
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))
    config = _config(trust="auto")
    db_path = _seed_chunk_profile(tmp_path, config)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)

    result = command_mod._chunk_backfill_run(
        config,
        db_path,
        apply=True,
        limit=10,
        retry_uncertain=False,
        policy="",
        confirm_raw_text=False,
    )

    assert isinstance(result, str)
    assert "status: refused" in result
    assert _CONSENT_ERROR in result
    assert provider.calls == []


def test_automatic_chunks_follow_honored_endpoint_and_keep_local_defaults():
    honored = _config(trust="self-hosted", base_url="http://10.2.10.10:4000/v1")
    automatic = _config(trust="auto", base_url="http://10.2.10.10:4000/v1")
    assert maintenance_mod._automatic_chunks_allowed(honored, command_mod) is True
    assert maintenance_mod._automatic_chunks_allowed(automatic, command_mod) is False

    fastembed = LCMConfig(embedding_provider="fastembed", embedding_model="model-a")
    assert maintenance_mod._automatic_chunks_allowed(fastembed, command_mod) is True
    ollama_loopback = LCMConfig(
        embedding_provider="ollama",
        embedding_model="model-a",
        ollama_base_url="http://127.0.0.1:11434",
    )
    assert maintenance_mod._automatic_chunks_allowed(ollama_loopback, command_mod) is True
    ollama_remote = LCMConfig(
        embedding_provider="ollama",
        embedding_model="model-a",
        ollama_base_url="http://remote.example:11434",
    )
    assert maintenance_mod._automatic_chunks_allowed(ollama_remote, command_mod) is False
    ollama_private_auto = LCMConfig(
        embedding_provider="ollama",
        embedding_model="model-a",
        ollama_base_url="http://10.2.10.10:11434",
        embedding_endpoint_trust="auto",
    )
    assert maintenance_mod._automatic_chunks_allowed(ollama_private_auto, command_mod) is False
    ollama_private_hosted = LCMConfig(
        embedding_provider="ollama",
        embedding_model="model-a",
        ollama_base_url="http://10.2.10.10:11434",
        embedding_endpoint_trust="self-hosted",
    )
    assert maintenance_mod._automatic_chunks_allowed(ollama_private_hosted, command_mod) is True


def test_privacy_revision_and_transform_match_auto_on_a_honored_endpoint():
    secret = "abcdefghijklmnop"
    raw = f"api_key={secret}"
    auto = _config(trust="auto")
    hosted = _config(trust="self-hosted")

    assert embedding_privacy_revision(auto) == embedding_privacy_revision(hosted)
    assert protect_embedding_text(raw, auto) == protect_embedding_text(raw, hosted)
    protected, revision, changed = protect_embedding_text(raw, hosted)
    assert changed is True
    assert secret not in protected
    assert revision.startswith("privacy:v3:")


def test_sensitive_pattern_status_adds_trust_keys_without_changing_existing_ones():
    auto = _config(trust="auto")
    hosted = _config(trust="self-hosted")
    refused = _config(trust="self-hosted", base_url="https://api.siliconflow.cn/v1")
    garbage = _config(trust="garbage")
    _reset_trust_warning()

    auto_status = sensitive_pattern_status(auto)
    hosted_status = sensitive_pattern_status(hosted)
    refused_status = sensitive_pattern_status(refused)
    garbage_status = sensitive_pattern_status(garbage)

    assert set(auto_status) == _PREEXISTING_STATUS_KEYS | {
        "embedding_endpoint_trust",
        "embedding_endpoint_trust_honored",
        "embedding_endpoint_trust_reason",
    }
    for status in (hosted_status, refused_status, garbage_status):
        assert set(status) == set(auto_status)
        for key in _PREEXISTING_STATUS_KEYS:
            assert status[key] == auto_status[key]

    assert auto_status["embedding_endpoint_trust"] == "auto"
    assert auto_status["embedding_endpoint_trust_honored"] is False
    assert auto_status["embedding_endpoint_trust_reason"] == "setting is auto"
    assert hosted_status["embedding_endpoint_trust"] == "self-hosted"
    assert hosted_status["embedding_endpoint_trust_honored"] is True
    assert hosted_status["embedding_endpoint_trust_reason"] == "10.2.10.10 is private"
    assert refused_status["embedding_endpoint_trust"] == "self-hosted"
    assert refused_status["embedding_endpoint_trust_honored"] is False
    assert (
        refused_status["embedding_endpoint_trust_reason"]
        == "hostname api.siliconflow.cn is not a literal address"
    )
    assert garbage_status["embedding_endpoint_trust"] == "auto"
    assert garbage_status["embedding_endpoint_trust_honored"] is False
    assert (
        garbage_status["embedding_endpoint_trust_reason"]
        == "unrecognized setting 'garbage'; treated as auto"
    )
    assert auto_status["embedding_privacy_revision"] == "active"
    assert hosted_status["embedding_privacy_enabled"] is True


def test_env_loads_endpoint_trust_without_normalizing_the_stored_value(monkeypatch):
    monkeypatch.delenv("LCM_EMBEDDING_ENDPOINT_TRUST", raising=False)
    assert LCMConfig.from_env().embedding_endpoint_trust == "auto"
    monkeypatch.setenv("LCM_EMBEDDING_ENDPOINT_TRUST", " Self-Hosted ")
    loaded = LCMConfig.from_env()
    assert loaded.embedding_endpoint_trust == " Self-Hosted "
    # No provider or endpoint is configured, so the normalized setting is refused.
    honored, detail = embedding_endpoint_trust_status(loaded)
    assert (honored, detail) == (False, "unknown provider")
    assert sensitive_pattern_status(loaded)["embedding_endpoint_trust"] == "self-hosted"
