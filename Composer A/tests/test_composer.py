import copy
import json
import os
import ssl
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composer


class ComposerTests(unittest.TestCase):
    def test_unc_cli_environment_aliases(self):
        settings = {'OPENAI_API_KEY': 'legacy-key', 'AZURE_OPENAI_API_KEY': 'azure-key',
                    'OPENAI_MODEL': 'legacy-model', 'AZURE_OPENAI_DEPLOYMENT': 'azure-model',
                    'OPENAI_ENDPOINT': 'https://legacy.example',
                    'AZURE_OPENAI_ENDPOINT': 'https://azureaiapi.cloud.unc.edu'}
        with patch.dict(os.environ, settings, clear=True), patch('composer.load_dotenv'), \
             patch('composer.compose', return_value=(composer.demo_score(1), [])) as generate, \
             patch('composer.export', return_value=Path('/tmp')):
            composer.main(['-p', 'piano', '-o', '/tmp', '--bars', '1'])
            self.assertEqual(generate.call_args.args[2:5],
                ('azure-model', 'azure-key', 'https://azureaiapi.cloud.unc.edu/openai/v1'))

    def test_native_macos_tls_fallback_keeps_key_off_arguments(self):
        import urllib.request
        import urllib.error
        error = urllib.error.URLError(ssl.SSLCertVerificationError('untrusted chain'))
        request = urllib.request.Request('https://example.com/responses', data=b'{}',
                                        headers={'Authorization': 'Bearer synthetic-key'})
        with patch('composer.sys.platform', 'darwin'), \
             patch('urllib.request.urlopen', side_effect=error), \
             patch('composer.subprocess.run') as native:
            native.return_value = subprocess.CompletedProcess([], 0, '{"output":[]}\n200', '')
            with composer.open_request(request) as response:
                self.assertEqual(json.load(response), {'output': []})
            self.assertNotIn('synthetic-key', ' '.join(native.call_args.args[0]))
            self.assertIn('synthetic-key', native.call_args.kwargs['input'])
            self.assertNotIn('--insecure', native.call_args.args[0])

    def test_unc_responses_api(self):
        import io
        response = {'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(composer.demo_score(1))}]}]}
        with patch('urllib.request.urlopen') as send:
            send.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
            result = composer.request_completion([{'role': 'user', 'content': 'JSON music'}],
                'openai/course-deployment', 'synthetic-course-key', 'https://azureaiapi.cloud.unc.edu/openai/v1')
            self.assertEqual(json.loads(result), composer.demo_score(1))
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, 'https://azureaiapi.cloud.unc.edu/openai/v1/responses')
            body = json.loads(request.data)
            self.assertEqual(body['model'], 'course-deployment')
            self.assertFalse(body['store'])
            self.assertFalse(body['background'])
            self.assertEqual(body['text']['format']['type'], 'json_object')

    def test_authentication_error_does_not_expose_key(self):
        import io
        import urllib.error
        error = urllib.error.HTTPError('https://example.com', 401, 'Unauthorized', {},
                                       io.BytesIO(b'{"error":{"message":"bad fake-secret"}}'))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'Authentication rejected') as raised:
                composer.request_completion([], 'test-model', 'fake-secret', 'https://example.com/v1')
        self.assertNotIn('fake-secret', str(raised.exception))

    def test_macos_missing_certificates_uses_system_bundle(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch('composer.sys.platform', 'darwin'), \
             patch('composer.ssl.create_default_context') as create, \
             patch('composer.Path.is_file', return_value=True):
            context = create.return_value
            context.get_ca_certs.return_value = []
            self.assertIs(composer.create_ssl_context(), context)
            context.load_verify_locations.assert_called_once_with(cafile='/etc/ssl/cert.pem')
            context.reset_mock()
            os.environ['SSL_CERT_FILE'] = '/custom/ca.pem'
            composer.create_ssl_context()
            context.load_verify_locations.assert_not_called()

    def test_ssl_context_verifies_certificates_and_hostnames(self):
        context = composer.create_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_dotenv_values_and_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            first, second = Path(tmp) / 'first.env', Path(tmp) / 'second.env'
            first.write_text('# settings\nexport OPENAI_API_KEY="fake-key" # comment\nOPENAI_MODEL=first\nUNRELATED=ignored\n')
            second.write_text('OPENAI_API_KEY=other\nOPENAI_MODEL=second\nOPENAI_BASE_URL=https://example.com/v1\n')
            os.environ['OPENAI_MODEL'] = 'exported'
            composer.load_dotenv([first, second])
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'fake-key')
            self.assertEqual(os.environ['OPENAI_MODEL'], 'exported')
            self.assertEqual(os.environ['OPENAI_BASE_URL'], 'https://example.com/v1')
            self.assertNotIn('UNRELATED', os.environ)

    def test_cli_uses_dotenv_key_and_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / '.env'
            path.write_text("OPENAI_API_KEY='fake-key'\nOPENAI_MODEL=dotenv-model\n")
            score = composer.demo_score(1)
            original = composer.load_dotenv
            with patch('composer.load_dotenv', side_effect=lambda: original([path])), \
                 patch('composer.compose', return_value=(score, [])) as generate, \
                 patch('composer.export', return_value=Path(tmp)):
                composer.main(['-p', 'piano', '-o', tmp, '--bars', '1'])
            self.assertEqual(generate.call_args.args[2:4], ('dotenv-model', 'fake-key'))

    def test_single_agent_joint_generation(self):
        calls = []
        def completion(messages):
            calls.append(copy.deepcopy(messages))
            return json.dumps(composer.demo_score(4))
        score, log = composer.compose('gentle piano', bars=4, completion=completion)
        self.assertEqual(len(calls), 1)
        self.assertEqual([m['role'] for m in calls[0]], ['system', 'user'])
        self.assertIn('melody, harmony', calls[0][0]['content'])
        self.assertEqual(len(log), 3)
        self.assertTrue(all(bar['melody'] and bar['harmony'] for bar in score['bars']))

    def test_repair_stays_in_same_conversation_and_is_bounded(self):
        calls = []
        def completion(messages):
            calls.append(copy.deepcopy(messages))
            return 'bad json' if len(calls) == 1 else json.dumps(composer.demo_score(1))
        composer.compose('piano', bars=1, completion=completion)
        self.assertEqual(len(calls[1]), 4)
        self.assertEqual(sum(m['role'] == 'system' for m in calls[1]), 1)
        self.assertIn('Validation failed', calls[1][-1]['content'])
        with self.assertRaisesRegex(ValueError, 'after 2 attempts'):
            composer.compose('piano', retries=1, completion=lambda _: '{}')

    def test_reject_invalid_music(self):
        mutations = [lambda s: s['bars'][0]['melody'][0].update(ticks=15),
                     lambda s: s['bars'][0]['harmony'][0].update(pitches=[]),
                     lambda s: s['bars'][0]['melody'][0].update(pitches=[60, 64]),
                     lambda s: s['bars'][0]['harmony'][0].update(pitches=[48, 48]),
                     lambda s: s.update(tempo=True),
                     lambda s: s.update(title='bad\nK:G'),
                     lambda s: s['bars'][0].update(chord='bad"label'),
                     lambda s: s['bars'][0]['melody'][0].update(pitches=[128]),
                     lambda s: s.update(bars=[])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                score = composer.demo_score(1)
                mutation(score)
                with self.assertRaises(ValueError):
                    composer.validate_score(score, 1)

    def test_http_request_and_response(self):
        class Response:
            def __enter__(self):
                import io
                return io.StringIO(json.dumps({'choices': [{'message': {'content': '{"ok":true}'}}]}))
            def __exit__(self, *args):
                pass
        with patch('urllib.request.urlopen', return_value=Response()) as mock:
            result = composer.request_completion([{'role': 'user', 'content': 'music'}], 'test-model', 'fake-key', 'https://example.com/v1')
            self.assertEqual(json.loads(result), {'ok': True})
            req = mock.call_args.args[0]
            self.assertEqual(req.full_url, 'https://example.com/v1/chat/completions')
            self.assertEqual(json.loads(req.data)['model'], 'test-model')

    def test_exports_and_midi_event_timing(self):
        score = composer.demo_score(4)
        # Include a rest, a repeated pitch, and an accidental to exercise encoders.
        score['bars'][0]['melody'] = [dict(pitches=[61], ticks=4), dict(pitches=[], ticks=4), dict(pitches=[61], ticks=8)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = composer.export(score, tmp, {'mode': 'test'})
            self.assertEqual(json.loads((folder / 'composition.json').read_text()), score)
            abc = (folder / 'composition.abc').read_text()
            self.assertIn('^C4 z4 ^C8', abc)
            self.assertIn('[=C,=E,=G,]16', abc)
            with wave.open(str(folder / 'composition.wav')) as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertAlmostEqual(audio.getnframes() / audio.getframerate(), 4 * 4 * 60 / score['tempo'], places=4)
                self.assertTrue(any(audio.readframes(audio.getnframes())))
            data = (folder / 'composition.mid').read_bytes()
            self.assertEqual(struct.unpack('>4sIHHH', data[:14]), (b'MThd', 6, 1, 2, 480))
            pos = 14
            for channel in range(2):
                self.assertEqual(data[pos:pos+4], b'MTrk')
                size = int.from_bytes(data[pos+4:pos+8], 'big')
                end = pos + 8 + size
                cursor, time, active, notes = pos + 8, 0, set(), 0
                while cursor < end:
                    delta = 0
                    while True:
                        byte = data[cursor]
                        cursor += 1
                        delta = (delta << 7) | (byte & 127)
                        if byte < 128:
                            break
                    time += delta
                    status = data[cursor]
                    cursor += 1
                    if status == 255:
                        kind, length = data[cursor:cursor+2]
                        cursor += 2 + length
                        if kind == 47:
                            self.assertFalse(active)
                            self.assertEqual(time, 4 * 4 * 480)
                    elif status & 240 == 192:
                        cursor += 1
                    else:
                        pitch, velocity = data[cursor:cursor+2]
                        cursor += 2
                        self.assertEqual(status & 15, channel)
                        if status & 240 == 144:
                            self.assertNotIn(pitch, active)
                            active.add(pitch)
                            notes += 1
                        else:
                            self.assertIn(pitch, active)
                            active.remove(pitch)
                self.assertGreater(notes, 0)
                pos = end
            self.assertEqual(pos, len(data))

    def test_cli_from_another_directory(self):
        script = Path(composer.__file__).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(script), '-p', 'piano', '-o', 'output', '--bars', '2', '--demo'], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list((Path(tmp) / 'output' / 'A_1').iterdir())), 5)

    def test_missing_key_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
            composer.compose('piano', api_key=None)


if __name__ == '__main__':
    unittest.main()
