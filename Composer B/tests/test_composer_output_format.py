"""Offline regressions for schema-constrained composition and failed replies."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from composer_b import agent as composer_b
from composer_b.composer_protocol import (
    COMPOSER_RESPONSE_SCHEMA,
    chat_response_format,
    responses_text_format,
)
from composer_b.agent import (
    InvalidComposerResponse,
    NegotiatingComposerAgent,
)
from composer_b.tests.test_negotiating_composer import response_json


class ComposerOutputFormatTests(unittest.TestCase):
    class DeepcopyableClient(httpx.Client):
        def __deepcopy__(self, memo):
            return self

    @staticmethod
    def sdk_reply(raw, *, responses):
        if responses:
            return {
                "id": "resp-schema-regression",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "test-deployment",
                "output": [{
                    "id": "msg-schema-regression",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw, "annotations": []}],
                }],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            }
        return {
            "id": "chatcmpl-schema-regression",
            "object": "chat.completion",
            "created": 0,
            "model": "test-deployment",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": raw},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    @staticmethod
    def composer(client, *, responses):
        entry = {
            "model": "test-deployment",
            "api_key": "offline-schema-secret",
            "base_url": "https://schema-test.invalid/openai/v1/" if responses else "https://schema-test.invalid/v1/",
            "http_client": client,
            "max_retries": 0,
            # The composer must enforce its protocol even when the supplied
            # configuration otherwise asks only for unconstrained JSON.
            "response_format": {"type": "json_object"},
            "price": [0, 0],
        }
        if responses:
            entry["model_client_cls"] = "ResponsesModelClient"
        return NegotiatingComposerAgent(llm_config={
            "config_list": [entry], "cache_seed": None, "max_retries": 0,
        })

    @staticmethod
    def snapshot(composer):
        return (
            composer.history, composer.negotiation_turn,
            composer.latest_response, composer.agreed_plan,
        )

    def test_schema_requires_complete_protocol_and_forbids_extra_fields(self):
        fields = {"message", "plan", "abc", "accepts_peer_plan"}
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["type"], "object")
        self.assertEqual(set(COMPOSER_RESPONSE_SCHEMA["properties"]), fields)
        self.assertEqual(set(COMPOSER_RESPONSE_SCHEMA["required"]), fields)
        self.assertIs(COMPOSER_RESPONSE_SCHEMA["additionalProperties"], False)
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["properties"]["message"]["type"], "string")
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["properties"]["abc"]["type"], "string")
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["properties"]["accepts_peer_plan"]["type"], "boolean")
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["properties"]["plan"]["type"], "array")
        self.assertEqual(COMPOSER_RESPONSE_SCHEMA["properties"]["plan"]["items"]["type"], "string")
        responses_format = responses_text_format()
        chat_format = chat_response_format()
        self.assertEqual(responses_format["type"], "json_schema")
        self.assertIs(responses_format["strict"], True)
        self.assertEqual(responses_format["schema"], COMPOSER_RESPONSE_SCHEMA)
        self.assertEqual(chat_format["type"], "json_schema")
        self.assertIs(chat_format["json_schema"]["strict"], True)
        self.assertEqual(chat_format["json_schema"]["schema"], COMPOSER_RESPONSE_SCHEMA)

    def test_real_responses_calls_send_strict_schema_for_every_operation(self):
        self.assert_schema_transmitted(responses=True)

    def test_real_chat_calls_send_strict_schema_for_every_operation(self):
        self.assert_schema_transmitted(responses=False)

    def assert_schema_transmitted(self, *, responses):
        requests = []

        def reply(request):
            requests.append(request)
            return httpx.Response(200, json=self.sdk_reply(response_json(), responses=responses), request=request)

        with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
            composer = self.composer(client, responses=responses)
            composer.compose("A jazz and blues inspired piano song.")
            composer.respond("Use a blues motif over a ii-V-I cadence.", turn=2)
            composer.finalize(("Use a blues motif and ii-V-I cadence in bars 13-16.",))

        self.assertEqual(len(requests), 3)
        for request in requests:
            body = json.loads(request.content)
            self.assertNotIn("offline-schema-secret", json.dumps(body))
            if responses:
                self.assertTrue(str(request.url).endswith("/openai/v1/responses"))
                self.assertEqual(body["text"]["format"], responses_text_format())
                self.assertNotIn("response_format", body)
            else:
                self.assertTrue(str(request.url).endswith("/v1/chat/completions"))
                self.assertEqual(body["response_format"], chat_response_format())
                self.assertNotIn("text", body)

    def test_ignored_schema_fails_once_and_preserves_existing_session(self):
        for responses in (True, False):
            with self.subTest(responses=responses):
                requests = []
                raw_replies = [response_json(), '{"abc": "incomplete protocol"}']

                def reply(request):
                    requests.append(request)
                    raw = raw_replies[len(requests) - 1]
                    return httpx.Response(200, json=self.sdk_reply(raw, responses=responses), request=request)

                with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
                    composer = self.composer(client, responses=responses)
                    composer.compose("Keep this established draft.")
                    before = self.snapshot(composer)
                    with self.assertRaises(InvalidComposerResponse) as caught:
                        composer.respond("A new peer proposal.", turn=3)
                    self.assertEqual(caught.exception.raw_response, raw_replies[1])
                    self.assertEqual(self.snapshot(composer), before)
                    self.assertEqual(len(requests), 2)

    def test_non_json_reply_is_retained_without_consuming_a_turn(self):
        composer = NegotiatingComposerAgent(llm_config=False)
        with patch.object(composer, "generate_reply", return_value=response_json()):
            composer.compose("A stable opening composition.")
        before = self.snapshot(composer)
        raw = "This reply ignores the required JSON format."
        with patch.object(composer, "generate_reply", return_value=raw) as generate:
            with self.assertRaises(InvalidComposerResponse) as caught:
                composer.respond("Add a contrasting bridge.", turn=2)
        generate.assert_called_once()
        self.assertEqual(caught.exception.raw_response, raw)
        self.assertEqual(self.snapshot(composer), before)

    def test_cli_saves_failed_reply_and_preserves_existing_composition_files(self):
        for existing in (False, True):
            with self.subTest(existing=existing), TemporaryDirectory(prefix="composer-failed-reply-") as directory:
                output = Path(directory) / "results"
                original_score = "A previously successful score.\n"
                original_record = '{"previous": "successful composition"}\n'
                if existing:
                    output.mkdir()
                    (output / "composition.abc").write_text(original_score, encoding="utf-8")
                    (output / "composition.json").write_text(original_record, encoding="utf-8")
                raw = '{"unexpected": "Keep this exact model reply for diagnosis."}'
                stderr = io.StringIO()
                # Bypass environment loading entirely; all generation is mocked.
                with patch.object(composer_b, "build_llm_config", return_value=(False, "test-deployment")):
                    with patch.object(NegotiatingComposerAgent, "generate_reply", return_value=raw) as generate:
                        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                            composer_b.main([
                                "--prompt", "A jazz and blues piano song.",
                                "--output-dir", str(output),
                            ])
                self.assertEqual(caught.exception.code, 1)
                generate.assert_called_once()
                diagnostic = output / "failed_response.txt"
                self.assertEqual(diagnostic.read_text(encoding="utf-8"), raw)
                self.assertIn(str(diagnostic), stderr.getvalue())
                if existing:
                    self.assertEqual((output / "composition.abc").read_text(encoding="utf-8"), original_score)
                    self.assertEqual((output / "composition.json").read_text(encoding="utf-8"), original_record)
                else:
                    self.assertFalse((output / "composition.abc").exists())
                    self.assertFalse((output / "composition.json").exists())


if __name__ == "__main__":
    unittest.main()
