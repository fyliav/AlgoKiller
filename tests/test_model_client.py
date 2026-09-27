import litellm
import pytest

from algokiller_harness import model_client


class _CloudflareLikeError(Exception):
    pass


def test_streaming_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("HARNESS_STREAM", raising=False)

    assert model_client.streaming_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled"])
def test_streaming_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("HARNESS_STREAM", value)

    assert model_client.streaming_enabled() is False


def test_request_timeout_uses_env_override(monkeypatch):
    monkeypatch.setenv("HARNESS_REQUEST_TIMEOUT_SECONDS", "42.5")

    assert model_client.request_timeout_seconds() == 42.5


def test_request_timeout_falls_back_on_invalid_value(monkeypatch):
    monkeypatch.setenv("HARNESS_REQUEST_TIMEOUT_SECONDS", "not-a-number")

    assert model_client.request_timeout_seconds() == model_client.DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_retry_delay_honours_server_retry_after(monkeypatch):
    exc = _CloudflareLikeError("error 524, 'retry_after': 120, 'retryable': True")

    delay = model_client._retry_delay_seconds(exc=exc, base_delay=1.0, attempt=1)

    assert delay == model_client.MAX_RETRY_DELAY_SECONDS


def test_retry_delay_backs_off_exponentially_and_caps():
    exc = _CloudflareLikeError("plain failure")

    delays = [model_client._retry_delay_seconds(exc=exc, base_delay=1.0, attempt=n) for n in range(1, 12)]

    assert delays[:3] == [1.0, 2.0, 4.0]
    assert max(delays) == model_client.MAX_RETRY_DELAY_SECONDS


def test_completion_with_retries_rebuilds_streamed_response(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return iter(
            [
                litellm.ModelResponse(
                    choices=[{"index": 0, "delta": {"role": "assistant", "content": "he"}}]
                ),
                litellm.ModelResponse(
                    choices=[{"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}]
                ),
            ]
        )

    monkeypatch.delenv("HARNESS_STREAM", raising=False)
    monkeypatch.setattr(litellm, "completion", fake_completion)

    response = model_client.completion_with_retries(
        max_attempts=1,
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert captured["stream"] is True
    assert captured["timeout"] == model_client.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert model_client.message_text(response.choices[0].message) == "hello"


def test_completion_with_retries_skips_streaming_when_disabled(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[{"index": 0, "message": {"role": "assistant", "content": "hello"}}]
        )

    monkeypatch.setenv("HARNESS_STREAM", "0")
    monkeypatch.setattr(litellm, "completion", fake_completion)

    response = model_client.completion_with_retries(
        max_attempts=1,
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "stream" not in captured
    assert model_client.message_text(response.choices[0].message) == "hello"


def test_completion_with_retries_retries_then_succeeds(monkeypatch):
    attempts = []

    def fake_completion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise _CloudflareLikeError("temporary failure")
        return litellm.ModelResponse(
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}}]
        )

    monkeypatch.setenv("HARNESS_STREAM", "0")
    monkeypatch.setattr(litellm, "completion", fake_completion)

    response = model_client.completion_with_retries(
        max_attempts=2,
        retry_delay_seconds=0,
        model="openai/gpt-test",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(attempts) == 2
    assert model_client.message_text(response.choices[0].message) == "ok"


def test_completion_with_retries_does_not_retry_auth_errors(monkeypatch):
    attempts = []

    def fake_completion(**kwargs):
        attempts.append(kwargs)
        raise _CloudflareLikeError("invalid api_key supplied")

    monkeypatch.setenv("HARNESS_STREAM", "0")
    monkeypatch.setattr(litellm, "completion", fake_completion)

    with pytest.raises(Exception):
        model_client.completion_with_retries(
            max_attempts=3,
            retry_delay_seconds=0,
            model="openai/gpt-test",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert len(attempts) == 1
