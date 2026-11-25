"""Offline tests for the full-song composer and its peer-message protocol.

Run from the repository root with:
    python -m unittest discover -s Composer B/tests -v

AutoGen must be installed, but all model calls are mocked.
"""

import json
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import autogen
import httpx

from composer_b.agent import (
    DEFAULT_MODEL,
    MAX_NEGOTIATION_TURNS,
    SYSTEM_MESSAGE,
    ComposerResponse,
    NegotiatingComposerAgent,
)


VALID_ABC = (
    "X:1\n"
    "T:Shared Horizons\n"
    "M:4/4\n"
    "L:1/8\n"
    "Q:1/4=100\n"
    "K:C\n"
    'V:1 clef=treble name="Melody"\n'
    "%%MIDI program 0\n"
    + "C2 E2 G2 c2 | " * 15
    + "G2 E2 C4 |]\n"
    'V:2 clef=bass name="Accompaniment"\n'
    "%%MIDI program 0\n"
    + "[C,E,G,]4 [G,,B,,D,]4 | " * 15
    + "[C,E,G,]8 |]\n"
)


def response_json(**changes):
    payload = {
        "message": "Use a lyrical melody over a restrained, syncopated accompaniment.",
        "plan": ["Use a lyrical stepwise melody.", "Use syncopated accompaniment."],
        "abc": VALID_ABC,
        "accepts_peer_plan": False,
    }
    payload.update(changes)
    return json.dumps(payload)


def all_content(messages):
    return "\n".join(str(message.get("content", "")) for message in messages)


class ComposerResponseTests(unittest.TestCase):
    def test_parses_typed_immutable_response(self):
        response = ComposerResponse.from_json(response_json())
        self.assertIsInstance(response, ComposerResponse)
        self.assertIsInstance(response.plan, tuple)
        self.assertEqual(len(response.plan), 2)
        self.assertEqual(response.abc, VALID_ABC.strip())
        self.assertIs(response.accepts_peer_plan, False)
        with self.assertRaises(FrozenInstanceError):
            response.message = "Changed"

    def test_rejects_malformed_json_and_non_object_payload(self):
        for raw in ("not JSON", "{", "[]", '"a response"', "null"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ComposerResponse.from_json(raw)

    def test_requires_every_response_field(self):
        for field in ("message", "plan", "abc", "accepts_peer_plan"):
            payload = json.loads(response_json())
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                ComposerResponse.from_json(json.dumps(payload))

    def test_rejects_empty_or_wrongly_typed_fields(self):
        invalid = (
            {"message": ""},
            {"message": "   "},
            {"message": []},
            {"plan": []},
            {"plan": "a plan"},
            {"plan": [""]},
            {"plan": ["   "]},
            {"plan": [1]},
            {"abc": ""},
            {"abc": None},
            {"accepts_peer_plan": "true"},
            {"accepts_peer_plan": 1},
            {"accepts_peer_plan": None},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ComposerResponse.from_json(response_json(**changes))

    def test_requires_all_abc_headers(self):
        for header in ("X:", "T:", "M:", "L:", "Q:", "K:"):
            abc = "\n".join(
                line for line in VALID_ABC.splitlines() if not line.startswith(header)
            )
            with self.subTest(header=header), self.assertRaises(ValueError):
                ComposerResponse.from_json(response_json(abc=abc))

    def test_rejects_melody_without_playable_accompaniment(self):
        single_voice = VALID_ABC.split("V:2", 1)[0]
        with self.assertRaises(ValueError):
            ComposerResponse.from_json(response_json(abc=single_voice))

    def test_rejects_empty_voice_or_missing_instrument(self):
        empty_voice = VALID_ABC.split("V:2", 1)[0] + "V:2\n%%MIDI program 0\n"
        missing_instrument = VALID_ABC.replace("%%MIDI program 0\n", "", 1)
        invalid_instruments = (
            VALID_ABC.replace("%%MIDI program 0", directive, 1)
            for directive in (
                "%%MIDI program 128",
                "%%MIDI program -1",
                "%%MIDI program 0 0",
                "%%MIDI program 17 0",
                "%%MIDI program 1 128",
            )
        )
        for abc in (empty_voice, missing_instrument, *invalid_instruments):
            with self.subTest(abc=abc), self.assertRaises(ValueError):
                ComposerResponse.from_json(response_json(abc=abc))


class NegotiatingComposerTests(unittest.TestCase):
    def setUp(self):
        self.agent = NegotiatingComposerAgent(
            name="ComposerA",
            preferences=("lyrical stepwise melody", "restrained accompaniment"),
            peer_preferences=("syncopated rhythm", "contrasting bridge"),
            llm_config=False,
        )
        self.reply_patch = patch.object(
            self.agent, "generate_reply", return_value=response_json()
        )
        self.generate_reply = self.reply_patch.start()
        self.addCleanup(self.reply_patch.stop)

    def snapshot(self):
        return (
            self.agent.history,
            self.agent.negotiation_turn,
            self.agent.latest_response,
            self.agent.agreed_plan,
        )

    def test_extends_real_autogen_assistant(self):
        self.assertIsInstance(self.agent, autogen.AssistantAgent)
        self.assertEqual(MAX_NEGOTIATION_TURNS, 6)

    def test_independent_generation_uses_one_model_call(self):
        result = self.agent.compose("Compose a hopeful 16-bar piano song.")
        self.assertIsInstance(result, ComposerResponse)
        self.generate_reply.assert_called_once()
        self.assertEqual(self.agent.negotiation_turn, 0)
        self.assertEqual(self.agent.latest_response, result)
        self.assertIsNone(self.agent.agreed_plan)
        content = all_content(self.generate_reply.call_args.kwargs["messages"])
        self.assertIn("hopeful 16-bar piano song", content)
        for preference in self.agent.preferences + self.agent.peer_preferences:
            self.assertIn(preference, content)

    def test_history_includes_composition_and_is_a_defensive_copy(self):
        self.agent.compose("A peaceful waltz.")
        before = self.agent.history
        content = all_content(before)
        self.assertIn("peaceful waltz", content)
        self.assertIn("Shared Horizons", content)
        copy = self.agent.history
        copy[0]["content"] = "tampered"
        copy.append({"role": "user", "content": "injected"})
        self.assertEqual(self.agent.history, before)

    def test_two_agent_instances_have_isolated_conversations(self):
        other = NegotiatingComposerAgent(
            name="ComposerB", preferences=("sustained harmony",), llm_config=False
        )
        other_history = other.history
        self.agent.compose("Only ComposerA should remember this.")
        self.assertEqual(other.history, other_history)
        self.assertIsNone(other.latest_response)
        self.assertEqual(other.negotiation_turn, 0)

    def test_peer_response_keeps_context_and_uses_global_turn_numbers(self):
        self.agent.compose("A jazz-inspired piano study.")
        for turn in (1, 3, 6):
            self.agent.respond(f"Peer proposal at global turn {turn}.", turn=turn)
            self.assertEqual(self.agent.negotiation_turn, turn)
        content = all_content(self.generate_reply.call_args.kwargs["messages"])
        self.assertIn("jazz-inspired piano study", content)
        self.assertIn("Peer proposal at global turn 1", content)
        self.assertIn("Peer proposal at global turn 3", content)
        self.assertIn("Peer proposal at global turn 6", content)
        self.assertIn("Shared Horizons", content)

    def test_structured_peer_response_is_communicated(self):
        self.agent.compose("An optimistic chamber piece.")
        peer = ComposerResponse.from_json(
            response_json(message="Please add a contrasting bridge.")
        )
        self.agent.respond(peer, turn=2)
        content = all_content(self.generate_reply.call_args.kwargs["messages"])
        self.assertIn(peer.message, content)
        self.assertIn(peer.plan[0], content)
        self.assertIn("Shared Horizons", content)

    def test_respond_requires_an_existing_composition(self):
        before = self.snapshot()
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.respond("Consider a slower tempo.", turn=1)
        self.assertEqual(self.snapshot(), before)
        self.generate_reply.assert_not_called()

    def test_rejects_invalid_global_turns_without_model_call(self):
        self.agent.compose("A short nocturne.")
        self.agent.respond("Try a softer opening.", turn=3)
        before = self.snapshot()
        calls = self.generate_reply.call_count
        for turn in (-1, 0, 1, 3, 7, True, 1.5, "4"):
            with self.subTest(turn=turn), self.assertRaises((ValueError, RuntimeError)):
                self.agent.respond("Another proposal.", turn=turn)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.generate_reply.call_count, calls)

    def test_sixth_global_turn_is_allowed_but_seventh_is_not(self):
        self.agent.compose("An energetic miniature.")
        self.agent.respond("This is the last peer turn.", turn=6)
        before = self.snapshot()
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.respond("One extra turn.", turn=7)
        self.assertEqual(self.snapshot(), before)

    def test_rejects_empty_peer_message(self):
        self.agent.compose("A folk dance.")
        before = self.snapshot()
        for message in ("", "   "):
            with self.subTest(message=message), self.assertRaises(ValueError):
                self.agent.respond(message, turn=1)
            self.assertEqual(self.snapshot(), before)

    def test_local_acceptance_does_not_end_negotiation(self):
        self.agent.compose("A duet with contrasting sections.")
        self.generate_reply.return_value = response_json(accepts_peer_plan=True)
        result = self.agent.respond("I suggest a contrasting bridge.", turn=2)
        self.assertIs(result.accepts_peer_plan, True)
        self.agent.respond("Confirm the final cadence too.", turn=4)
        self.assertEqual(self.agent.negotiation_turn, 4)

    def test_finalize_requires_composition_and_explicit_plan(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.finalize(("Use a lyrical melody.",))
        self.generate_reply.assert_not_called()
        self.agent.compose("A song to negotiate.")
        before = self.snapshot()
        for plan in ((), [], [""], ["  "], "Use a lyrical melody."):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.agent.finalize(plan)
            self.assertEqual(self.snapshot(), before)

    def test_finalize_sends_agreed_plan_then_locks_session(self):
        self.agent.compose("A reflective piano song.")
        agreed = ("Keep a stepwise melody.", "Add syncopation only in the bridge.")
        result = self.agent.finalize(agreed)
        content = all_content(self.generate_reply.call_args.kwargs["messages"])
        for requirement in agreed:
            self.assertIn(requirement, content)
        self.assertEqual(self.agent.latest_response, result)
        self.assertEqual(self.agent.agreed_plan, agreed)
        self.assertEqual(result.plan, ComposerResponse.from_json(response_json()).plan)
        before = self.snapshot()
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.respond("Change the bridge again.", turn=1)
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.finalize(agreed)
        self.assertEqual(self.snapshot(), before)

    def test_new_composition_resets_session_and_discards_old_context(self):
        self.agent.compose("Original composition prompt.")
        self.agent.respond("Original peer critique.", turn=5)
        self.agent.finalize(("Original agreed plan.",))
        self.agent.compose("Replacement composition prompt.")
        self.assertEqual(self.agent.negotiation_turn, 0)
        self.assertIsNone(self.agent.agreed_plan)
        content = all_content(self.agent.history)
        self.assertIn("Replacement composition prompt", content)
        self.assertNotIn("Original composition prompt", content)
        self.assertNotIn("Original peer critique", content)
        self.agent.respond("A new peer proposal.", turn=1)
        self.assertEqual(self.agent.negotiation_turn, 1)

    def test_failed_compose_does_not_reset_existing_session(self):
        self.agent.compose("Keep this composition.")
        self.agent.respond("Keep this peer critique.", turn=3)
        before = self.snapshot()
        self.generate_reply.return_value = "invalid JSON"
        with self.assertRaises(ValueError):
            self.agent.compose("This composition attempt fails.")
        self.assertEqual(self.snapshot(), before)

    def test_failed_peer_response_does_not_consume_turn(self):
        self.agent.compose("A carefully paced ballad.")
        before = self.snapshot()
        self.generate_reply.return_value = response_json(abc="not a score")
        with self.assertRaises(ValueError):
            self.agent.respond("A proposal worth retrying.", turn=2)
        self.assertEqual(self.snapshot(), before)
        self.generate_reply.return_value = response_json()
        self.agent.respond("A proposal worth retrying.", turn=2)
        self.assertEqual(self.agent.negotiation_turn, 2)

    def test_failed_finalize_does_not_lock_session(self):
        self.agent.compose("A piano prelude.")
        before = self.snapshot()
        self.generate_reply.return_value = "invalid JSON"
        with self.assertRaises(ValueError):
            self.agent.finalize(("End on the tonic.",))
        self.assertEqual(self.snapshot(), before)
        self.generate_reply.return_value = response_json()
        self.agent.respond("Clarify the cadence.", turn=1)
        self.assertEqual(self.agent.negotiation_turn, 1)

    def test_api_failure_preserves_state_for_every_operation(self):
        self.agent.compose("A stable starting composition.")
        operations = (
            lambda: self.agent.compose("An unavailable replacement."),
            lambda: self.agent.respond("An unavailable peer reply.", turn=2),
            lambda: self.agent.finalize(("Keep the original cadence.",)),
        )
        for operation in operations:
            before = self.snapshot()
            with self.subTest(operation=operation):
                with patch.object(
                    self.agent, "generate_reply", side_effect=RuntimeError("API unavailable")
                ):
                    with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                        operation()
                self.assertEqual(self.snapshot(), before)


class AutoGenTransportIntegrationTests(unittest.TestCase):
    def test_real_autogen_and_openai_path_supports_repeated_trials_and_peer_context(self):
        """Exercise real request assembly and parsing without reaching a server."""

        class DeepcopyableClient(httpx.Client):
            def __deepcopy__(self, memo):
                return self

        requests = []

        def model_reply(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-offline-composer-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": DEFAULT_MODEL,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_json()},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                },
                request=request,
            )

        with DeepcopyableClient(transport=httpx.MockTransport(model_reply)) as client:
            agent = NegotiatingComposerAgent(
                name="IntegrationComposer",
                preferences=("lyrical melody",),
                peer_preferences=("syncopated accompaniment",),
                llm_config={
                    "config_list": [{
                        "model": DEFAULT_MODEL,
                        "api_key": "offline-test-key",
                        "base_url": "https://composer.invalid/v1",
                        "http_client": client,
                        "price": [0, 0],
                    }],
                    "max_retries": 0,
                },
            )
            self.assertIsNone(agent.llm_config["cache_seed"])
            # More than six initial trials must work on the same agent. These
            # calls are independent compositions, not negotiation turns.
            for index in range(8):
                result = agent.compose(f"Independent trial number {index}.")
                self.assertIsInstance(result, ComposerResponse)
                self.assertEqual(agent.negotiation_turn, 0)
            agent.respond("Keep the melody; syncopate bars 9-12.", turn=2)
            final = agent.finalize(("Syncopate accompaniment in bars 9-12.",))
            self.assertNotIn("offline-test-key", json.dumps(agent.history))
            self.assertNotIn("offline-test-key", final.to_json())

        self.assertEqual(DEFAULT_MODEL, "gpt-6.1-luna")
        self.assertEqual(len(requests), 10)
        for request in requests:
            self.assertEqual(request["model"], "gpt-6.1-luna")
            self.assertNotIn("offline-test-key", json.dumps(request))
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertEqual(request["messages"][0]["content"], SYSTEM_MESSAGE)
        self.assertIn("lyrical melody", all_content(requests[0]["messages"]))
        self.assertIn("syncopated accompaniment", all_content(requests[0]["messages"]))
        last_trial = all_content(requests[7]["messages"])
        self.assertIn("Independent trial number 7", last_trial)
        self.assertNotIn("Independent trial number 6", last_trial)
        peer_context = all_content(requests[8]["messages"])
        self.assertIn("Independent trial number 7", peer_context)
        self.assertIn("Shared Horizons", peer_context)
        self.assertIn("Keep the melody; syncopate bars 9-12", peer_context)
        self.assertIn("GLOBAL negotiation turn 2/6", peer_context)
        self.assertIn(
            "Syncopate accompaniment in bars 9-12", all_content(requests[9]["messages"])
        )
        self.assertEqual(agent.agreed_plan, ("Syncopate accompaniment in bars 9-12.",))
        self.assertEqual(final.plan, ComposerResponse.from_json(response_json()).plan)


if __name__ == "__main__":
    unittest.main()
