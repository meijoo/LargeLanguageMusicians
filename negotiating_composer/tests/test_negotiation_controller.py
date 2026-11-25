import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from ComposerSingle import composer
from negotiating_composer.agent import ComposerResponse
from negotiating_composer.negotiate import run


class ControllerTests(unittest.TestCase):
    def exercise(self, accepts, different_plan=False, full_six_turns=False, withdraw_at_end=False):
        score = composer.demo_score(16)
        plan = ('Use a 16-bar piano blues with a final tonic cadence.',)
        response = ComposerResponse('Peer proposal.', plan, composer.to_abc(score), accepts)
        peer = Mock(history=[])
        peer.compose.return_value = ComposerResponse('Initial proposal.', plan, response.abc, False)
        peer.respond.return_value = (ComposerResponse('Changed plan.', ('Use a different harmony.',), response.abc, True)
                                      if different_plan else response)
        if withdraw_at_end:
            peer.respond.side_effect = [response, response, ComposerResponse(
                'I withdraw acceptance.', plan, response.abc, False)]
        wrapper = json.dumps(dict(message='My response.', plan=plan, score=score, accepts_peer_plan=accepts))
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-secret', 'OPENAI_MODEL': 'test-model'}, clear=True), \
                patch('negotiating_composer.negotiate.single.load_dotenv'), \
                patch('negotiating_composer.negotiate.NegotiatingComposerAgent', return_value=peer), \
                patch('negotiating_composer.negotiate.single.compose', return_value=(score, [{'role': 'system', 'content': composer.SYSTEM_PROMPT}])), \
                patch('negotiating_composer.negotiate.single.request_completion', return_value=wrapper), \
                patch('negotiating_composer.negotiate.single.export') as export:
            destination = Path(directory)
            if accepts and not different_plan and not withdraw_at_end:
                run('jazz blues piano', destination, full_six_turns=full_six_turns)
                self.assertEqual(export.call_count, 2)
            else:
                with self.assertRaises(RuntimeError):
                    run('jazz blues piano', destination, full_six_turns=full_six_turns)
                self.assertEqual(export.call_count, 1)
            raw = (destination / 'negotiation.json').read_text()
            self.assertNotIn('test-secret', raw)
            return json.loads(raw)

    def test_explicit_mutual_agreement_exports_final(self):
        result = self.exercise(True)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['exchanges']), 5)

    def test_no_agreement_does_not_export_final(self):
        result = self.exercise(False)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['exchanges']), 8)

    def test_full_six_turns_continues_after_early_agreement(self):
        result = self.exercise(True, full_six_turns=True)
        self.assertEqual(result['status'], 'complete')
        turns = [entry for entry in result['exchanges'] if entry['phase'].startswith('turn ')]
        self.assertEqual([entry['phase'] for entry in turns], [f'turn {n}' for n in range(1, 7)])
        self.assertEqual([entry['speaker'] for entry in turns], ['ComposerSingle', 'NegotiatingComposer'] * 3)

    def test_acceptance_flags_with_different_plans_are_not_agreement(self):
        result = self.exercise(True, different_plan=True)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('agreed_plan', result)

    def test_withdrawal_at_turn_six_clears_earlier_agreement(self):
        result = self.exercise(True, full_six_turns=True, withdraw_at_end=True)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('agreed_plan', result)


if __name__ == '__main__':
    unittest.main()
