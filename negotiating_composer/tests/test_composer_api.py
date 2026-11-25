"""Offline configuration and UNC/Azure Responses transport checks."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from negotiating_composer.composer_api import DEFAULT_MODEL, build_llm_config
from negotiating_composer.agent import (
    ComposerResponse,
    NegotiatingComposerAgent,
    SYSTEM_MESSAGE,
)
from negotiating_composer.tests.test_negotiating_composer import response_json


class ComposerConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="composer-config-")
        self.addCleanup(self.directory.cleanup)
        self.env_file = Path(self.directory.name) / ".env"
        self.env_file.write_text("", encoding="utf-8")

    def build(self, environment, model=None):
        with patch.dict(os.environ, environment, clear=True):
            return build_llm_config(model, env_file=self.env_file)

    @staticmethod
    def entry(config):
        return config["config_list"][0]

    def test_loads_supplied_dotenv_and_uses_unc_responses_route(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=unc-subscription-test-key\n"
            "OPENAI_MODEL=gpt-5.6-luna\n"
            "OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu\n",
            encoding="utf-8",
        )
        config, model = self.build({})
        entry = self.entry(config)
        self.assertEqual(model, "gpt-5.6-luna")
        self.assertEqual(entry["model"], model)
        self.assertEqual(entry["api_key"], "unc-subscription-test-key")
        self.assertEqual(entry["base_url"], "https://azureaiapi.cloud.unc.edu/openai/v1/")
        self.assertEqual(entry["model_client_cls"], "ResponsesModelClient")
        self.assertIsNone(config["cache_seed"])

    def test_dotenv_does_not_override_exported_values(self):
        self.env_file.write_text(
            "OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-deployment\n"
            "OPENAI_ENDPOINT=https://file.example\n",
            encoding="utf-8",
        )
        config, model = self.build({
            "OPENAI_API_KEY": "exported-key",
            "OPENAI_MODEL": "exported-deployment",
            "OPENAI_ENDPOINT": "https://exported.example",
        })
        entry = self.entry(config)
        self.assertEqual(entry["api_key"], "exported-key")
        self.assertEqual(model, "exported-deployment")
        self.assertEqual(entry["base_url"], "https://exported.example/openai/v1/")

    def test_azure_aliases_take_precedence_over_generic_names(self):
        config, model = self.build({
            "OPENAI_API_KEY": "generic-key",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "OPENAI_MODEL": "generic-model",
            "AZURE_OPENAI_DEPLOYMENT": "azure-deployment",
            "OPENAI_ENDPOINT": "https://generic.example",
            "AZURE_OPENAI_ENDPOINT": "https://azure.example",
            "OPENAI_BASE_URL": "https://stale.example/v1/",
        })
        entry = self.entry(config)
        self.assertEqual(entry["api_key"], "azure-key")
        self.assertEqual(model, "azure-deployment")
        self.assertEqual(entry["base_url"], "https://azure.example/openai/v1/")

    def test_explicit_model_overrides_both_environment_model_names(self):
        config, model = self.build({
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_ENDPOINT": "https://azure.example",
            "AZURE_OPENAI_DEPLOYMENT": "azure-deployment",
            "OPENAI_MODEL": "generic-model",
        }, model="run-specific-alias")
        self.assertEqual(model, "run-specific-alias")
        self.assertEqual(self.entry(config)["model"], "run-specific-alias")

    def test_blank_azure_aliases_allow_generic_configuration(self):
        config, model = self.build({
            "AZURE_OPENAI_API_KEY": "  ",
            "AZURE_OPENAI_DEPLOYMENT": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "OPENAI_API_KEY": "generic-key",
            "OPENAI_MODEL": "generic-alias",
            "OPENAI_ENDPOINT": "https://generic.example",
        })
        entry = self.entry(config)
        self.assertEqual(model, "generic-alias")
        self.assertEqual(entry["api_key"], "generic-key")
        self.assertEqual(entry["base_url"], "https://generic.example/openai/v1/")

    def test_accepts_origin_or_existing_v1_suffix_without_duplication(self):
        for endpoint in (
            "https://azureaiapi.cloud.unc.edu",
            "https://azureaiapi.cloud.unc.edu/",
            "https://azureaiapi.cloud.unc.edu/openai/v1",
            "https://azureaiapi.cloud.unc.edu/openai/v1/",
        ):
            with self.subTest(endpoint=endpoint):
                config, _ = self.build({
                    "OPENAI_API_KEY": "unc-key",
                    "OPENAI_MODEL": "gpt-5.6-luna",
                    "OPENAI_ENDPOINT": endpoint,
                })
                self.assertEqual(
                    self.entry(config)["base_url"],
                    "https://azureaiapi.cloud.unc.edu/openai/v1/",
                )

    def test_rejects_ambiguous_or_invalid_endpoint_without_echoing_it(self):
        invalid = (
            "azureaiapi.cloud.unc.edu",
            "ftp://azureaiapi.cloud.unc.edu",
            "https://azureaiapi.cloud.unc.edu/v1",
            "https://azureaiapi.cloud.unc.edu/openai/v1/responses",
            "https://azureaiapi.cloud.unc.edu?api-key=secret-in-url",
            "https://azureaiapi.cloud.unc.edu/#secret-in-url",
            "https://user:secret-in-url@azureaiapi.cloud.unc.edu",
        )
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError) as caught:
                self.build({
                    "OPENAI_API_KEY": "unc-key",
                    "OPENAI_MODEL": "gpt-5.6-luna",
                    "OPENAI_ENDPOINT": endpoint,
                })
            self.assertNotIn("secret-in-url", str(caught.exception))

    def test_endpoint_requires_an_explicit_deployment_instead_of_public_default(self):
        for environment in (
            {
                "OPENAI_API_KEY": "unc-key",
                "OPENAI_ENDPOINT": "https://azureaiapi.cloud.unc.edu",
            },
            {
                "AZURE_OPENAI_API_KEY": "azure-key",
                "OPENAI_BASE_URL": "https://provider.example/v1/",
            },
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.build(environment)

    def test_azure_credentials_without_endpoint_cannot_fall_back_to_public_openai(self):
        for environment in (
            {"AZURE_OPENAI_API_KEY": "azure-key"},
            {"OPENAI_API_KEY": "generic-key", "AZURE_OPENAI_DEPLOYMENT": "alias"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.build(environment)

    def test_legacy_base_url_keeps_chat_completions_configuration(self):
        config, model = self.build({
            "OPENAI_API_KEY": "provider-test-key",
            "OPENAI_MODEL": "provider-alias",
            "OPENAI_BASE_URL": "https://provider.example/v1/",
        })
        entry = self.entry(config)
        self.assertEqual(model, "provider-alias")
        self.assertEqual(entry["base_url"], "https://provider.example/v1/")
        self.assertNotIn("model_client_cls", entry)

    def test_azure_aliases_can_use_an_explicit_legacy_base_url(self):
        config, model = self.build({
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_DEPLOYMENT": "azure-alias",
            "OPENAI_BASE_URL": "https://provider.example/v1/",
        })
        entry = self.entry(config)
        self.assertEqual(model, "azure-alias")
        self.assertEqual(entry["api_key"], "azure-key")
        self.assertEqual(entry["base_url"], "https://provider.example/v1/")
        self.assertNotIn("model_client_cls", entry)

    def test_public_openai_retains_default_and_environment_model_support(self):
        config, model = self.build({"OPENAI_API_KEY": "public-test-key"})
        self.assertEqual(model, DEFAULT_MODEL)
        self.assertNotIn("model_client_cls", self.entry(config))
        _, environment_model = self.build({
            "OPENAI_API_KEY": "public-test-key", "OPENAI_MODEL": "public-alias",
        })
        self.assertEqual(environment_model, "public-alias")

    def test_missing_api_key_fails_before_creating_a_client(self):
        for environment in ({}, {"OPENAI_API_KEY": "", "AZURE_OPENAI_API_KEY": ""}):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.build(environment)


class ResponsesTransportTests(unittest.TestCase):
    class DeepcopyableClient(httpx.Client):
        def __deepcopy__(self, memo):
            return self

    @staticmethod
    def sdk_response(*, status="completed", text=None):
        if text is None:
            text = response_json()
        return {
            "id": "resp-offline-composer-test",
            "object": "response",
            "created_at": 0,
            "status": status,
            "model": "gpt-5.6-luna",
            "output": [{
                "id": "msg-offline-test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }],
            "usage": {"input_tokens": 20, "output_tokens": 40, "total_tokens": 60},
        }

    def agent(self, client):
        return NegotiatingComposerAgent(
            name="UNCComposer",
            preferences=("jazz harmony",),
            peer_preferences=("blues melody",),
            llm_config={
                "config_list": [{
                    "model": "gpt-5.6-luna",
                    "api_key": "unc-offline-secret",
                    "base_url": "https://azureaiapi.cloud.unc.edu/openai/v1/",
                    "model_client_cls": "ResponsesModelClient",
                    "http_client": client,
                    "max_retries": 0,
                }],
                "max_retries": 0,
                "cache_seed": None,
            },
        )

    def test_real_autogen_responses_path_preserves_context_and_uses_unc_auth(self):
        requests = []

        def reply(request):
            requests.append(request)
            return httpx.Response(200, json=self.sdk_response(), request=request)

        with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
            composer = self.agent(client)
            draft = composer.compose("A jazz and blues piano song.")
            composer.respond("Keep the blues melody; add a ii-V-I cadence.", turn=2)
            final = composer.finalize(("Use a ii-V-I cadence in bars 15-16.",))
            self.assertIsInstance(draft, ComposerResponse)
            self.assertIsInstance(final, ComposerResponse)
            self.assertNotIn("unc-offline-secret", json.dumps(composer.history))
            self.assertNotIn("unc-offline-secret", final.to_json())

        self.assertEqual(len(requests), 3)
        bodies = []
        for request in requests:
            self.assertEqual(
                str(request.url),
                "https://azureaiapi.cloud.unc.edu/openai/v1/responses",
            )
            self.assertEqual(request.headers["api-key"], "unc-offline-secret")
            self.assertEqual(request.headers["authorization"], "Bearer unc-offline-secret")
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(body["model"], "gpt-5.6-luna")
            self.assertIs(body["store"], False)
            self.assertNotIn("unc-offline-secret", json.dumps(body))
            self.assertNotIn("messages", body)
            self.assertEqual(body["input"][0]["role"], "system")
            self.assertEqual(body["input"][0]["content"], SYSTEM_MESSAGE)
        initial = json.dumps(bodies[0]["input"])
        self.assertIn("jazz harmony", initial)
        self.assertIn("blues melody", initial)
        self.assertIn("jazz and blues piano song", initial)
        peer = json.dumps(bodies[1]["input"])
        self.assertIn("jazz and blues piano song", peer)
        self.assertIn("ii-V-I cadence", peer)
        self.assertIn("Shared Horizons", peer)
        self.assertIn("bars 15-16", json.dumps(bodies[2]["input"]))

    def test_incomplete_responses_fail_without_committing_history(self):
        def reply(request):
            return httpx.Response(
                200, json=self.sdk_response(status="incomplete"), request=request,
            )

        with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
            composer = self.agent(client)
            with self.assertRaises((ValueError, RuntimeError)):
                composer.compose("A jazz piano song.")
            self.assertEqual(composer.history, [])
            self.assertIsNone(composer.latest_response)

    def test_empty_responses_fail_without_committing_history(self):
        def reply(request):
            return httpx.Response(200, json=self.sdk_response(text=""), request=request)

        with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
            composer = self.agent(client)
            with self.assertRaises((ValueError, RuntimeError)):
                composer.compose("A jazz piano song.")
            self.assertEqual(composer.history, [])
            self.assertIsNone(composer.latest_response)

    def test_multiple_responses_config_entries_are_all_registered(self):
        def reply(request):
            return httpx.Response(200, json=self.sdk_response(), request=request)

        with self.DeepcopyableClient(transport=httpx.MockTransport(reply)) as client:
            first = self.agent(client)
            config = dict(first.llm_config)
            entry = dict(config["config_list"][0])
            config["config_list"] = [entry, dict(entry, model="second-enabled-alias")]
            composer = NegotiatingComposerAgent(llm_config=config)
            result = composer.compose("A jazz and blues piano song.")
            self.assertIsInstance(result, ComposerResponse)


if __name__ == "__main__":
    unittest.main()
