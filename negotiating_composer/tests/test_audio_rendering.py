"""Offline WAV export tests; no API calls or audio-device access."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import numpy as np

from negotiating_composer import agent as negotiating_composer, render_audio
from negotiating_composer.agent import NegotiatingComposerAgent
from negotiating_composer.tests.test_negotiating_composer import response_json


SHORT_ABC = """X:1
T:Two Voice Audio Test
M:4/4
L:1/8
Q:1/4=120
K:C
V:1 name="Melody"
%%MIDI program 0
V:2 name="Guitar Accompaniment"
%%MIDI program 24
V:1
"C7"C2 E2 G2 c2 | "G7"d2 c2 B2 G2 |]
V:2
"C7"C,2 G,2 E,2 G,2 | "G7"G,,2 D,2 F,2 D,2 |]
"""


class AudioRenderingTests(unittest.TestCase):
    def test_real_export_preserves_voices_notes_programs_and_source(self):
        from music21 import midi

        with TemporaryDirectory(prefix="composer-render-") as directory:
            root = Path(directory)
            source = root / "draft.abc"
            source.write_text(SHORT_ABC, encoding="utf-8")
            original = source.read_bytes()
            destination = root / "audio"
            result = render_audio.render_abc_to_wav(source, destination)

            self.assertEqual(result.resolve(), (destination / "draft.wav").resolve())
            self.assertEqual(source.read_bytes(), original)
            with wave.open(str(result), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 2)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), 44100)
                # Two 4/4 measures at 120 BPM plus a two-second release tail.
                duration = audio.getnframes() / audio.getframerate()
                self.assertAlmostEqual(duration, 6.0, delta=0.15)
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            peak = int(np.max(np.abs(samples.astype(np.int32))))
            self.assertGreater(peak, 100, "Rendered WAV must contain audible samples.")
            self.assertLess(peak, 32760, "The short test score must not clip.")

            midi_file = midi.MidiFile()
            midi_file.open(str(destination / "draft.mid"))
            try:
                midi_file.read()
            finally:
                midi_file.close()
            note_counts = []
            programs = set()
            for track in midi_file.tracks:
                notes = [event for event in track.events
                         if event.type == midi.ChannelVoiceMessages.NOTE_ON and event.velocity]
                if notes:
                    note_counts.append(len(notes))
                programs.update(event.data for event in track.events
                                if event.type == midi.ChannelVoiceMessages.PROGRAM_CHANGE)
            # Chord labels are annotations, not extra synthesized accompaniment.
            self.assertEqual(sorted(note_counts), [8, 8])
            self.assertEqual(programs, {0, 24})

    def test_same_instrument_voices_keep_independent_midi_channels(self):
        from music21 import midi

        with TemporaryDirectory(prefix="composer-piano-voices-") as directory:
            source = Path(directory) / "two_pianos.abc"
            source.write_text(SHORT_ABC.replace("%%MIDI program 24", "%%MIDI program 0"), encoding="utf-8")
            render_audio.render_abc_to_wav(source)
            midi_file = midi.MidiFile()
            midi_file.open(str(source.with_suffix(".mid")))
            try:
                midi_file.read()
            finally:
                midi_file.close()
            voice_channels, note_counts = [], []
            for track in midi_file.tracks:
                notes = [event for event in track.events
                         if event.type == midi.ChannelVoiceMessages.NOTE_ON and event.velocity]
                if notes:
                    voice_channels.append({event.channel for event in notes})
                    note_counts.append(len(notes))
            self.assertEqual(sorted(note_counts), [8, 8])
            self.assertEqual(len(voice_channels), 2)
            self.assertTrue(all(len(channels) == 1 for channels in voice_channels))
            # Sharing an instrument must not let one voice's note-off messages
            # cut short another voice's overlapping notes of the same pitch.
            self.assertEqual(len(set.union(*voice_channels)), 2)
            self.assertNotIn(10, set.union(*voice_channels), "Melodic voices must avoid the drum channel.")

    def test_global_headers_after_voice_declarations_apply_to_every_voice(self):
        from music21 import midi

        abc = """X:1
T:Global Header Test
V:1 name="Melody"
%%MIDI program 0
V:2 name="Accompaniment"
%%MIDI program 24
M:4/4
L:1/8
Q:1/4=120
K:G
V:1
F2 G2 A2 B2 |]
V:2
G,8 |]
"""
        with TemporaryDirectory(prefix="composer-global-headers-") as directory:
            source = Path(directory) / "global_headers.abc"
            source.write_text(abc, encoding="utf-8")
            result = render_audio.render_abc_to_wav(source)
            self.assertEqual(source.read_text(encoding="utf-8"), abc)
            with wave.open(str(result), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                # One shared 4/4 bar at 120 BPM plus the release tail.
                self.assertAlmostEqual(duration, 4.0, delta=0.15)
            midi_file = midi.MidiFile()
            midi_file.open(str(source.with_suffix(".mid")))
            try:
                midi_file.read()
            finally:
                midi_file.close()
            voices = []
            for track in midi_file.tracks:
                notes = [event for event in track.events
                         if event.type == midi.ChannelVoiceMessages.NOTE_ON and event.velocity]
                if notes:
                    voices.append(notes)
            self.assertEqual(sorted(len(notes) for notes in voices), [1, 4])
            melody = next(notes for notes in voices if len(notes) == 4)
            self.assertEqual(melody[0].pitch, 66, "The global G-major key must make F sharp.")

    def test_trailing_rest_bars_remain_in_the_rendered_duration(self):
        from music21 import midi

        abc = SHORT_ABC.replace(
            '"C7"C2 E2 G2 c2 | "G7"d2 c2 B2 G2 |]', '"C7"C8 | z8 | z8 |]',
        ).replace(
            '"C7"C,2 G,2 E,2 G,2 | "G7"G,,2 D,2 F,2 D,2 |]', '"C7"C,8 | z8 | z8 |]',
        )
        with TemporaryDirectory(prefix="composer-trailing-rests-") as directory:
            source = Path(directory) / "rest_ending.abc"
            source.write_text(abc, encoding="utf-8")
            result = render_audio.render_abc_to_wav(source)
            self.assertEqual(source.read_text(encoding="utf-8"), abc)
            with wave.open(str(result), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                # Three bars at 120 BPM last six seconds even though the last
                # sounding note ends after the first bar; retain the release tail.
                self.assertAlmostEqual(duration, 8.0, delta=0.15)
            midi_file = midi.MidiFile()
            midi_file.open(str(source.with_suffix(".mid")))
            try:
                midi_file.read()
            finally:
                midi_file.close()
            note_counts = []
            for track in midi_file.tracks:
                notes = [event for event in track.events
                         if event.type == midi.ChannelVoiceMessages.NOTE_ON and event.velocity]
                if notes:
                    note_counts.append(len(notes))
            self.assertEqual(note_counts, [1, 1], "Rest bars must not add synthesized notes.")

    def test_empty_or_all_rest_score_does_not_publish_or_replace_audio(self):
        silent = SHORT_ABC.replace(
            '"C7"C2 E2 G2 c2 | "G7"d2 c2 B2 G2 |]', "z8 | z8 |]",
        ).replace(
            '"C7"C,2 G,2 E,2 G,2 | "G7"G,,2 D,2 F,2 D,2 |]', "z8 | z8 |]",
        )
        for abc in ("", silent):
            with self.subTest(abc=abc), TemporaryDirectory(prefix="composer-silent-") as directory:
                source = Path(directory) / "draft.abc"
                source.write_text(abc, encoding="utf-8")
                previous = source.with_suffix(".wav")
                previous.write_bytes(b"previous valid audio must survive")
                with self.assertRaises((ValueError, RuntimeError)):
                    render_audio.render_abc_to_wav(source)
                self.assertEqual(previous.read_bytes(), b"previous valid audio must survive")


class AutomaticAudioCliTests(unittest.TestCase):
    @staticmethod
    def fake_render(abc_path, output_dir=None, **kwargs):
        source = Path(abc_path)
        destination = Path(output_dir) if output_dir is not None else source.parent
        destination.mkdir(parents=True, exist_ok=True)
        wav_path = destination / source.with_suffix(".wav").name
        with wave.open(str(wav_path), "wb") as audio:
            audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x00" * 2 * 441)
        (destination / source.with_suffix(".mid").name).write_bytes(b"mock MIDI")
        return wav_path

    def test_default_cli_saves_score_record_midi_and_wav(self):
        with TemporaryDirectory(prefix="composer-cli-wav-") as directory:
            output = Path(directory) / "result"
            stdout = io.StringIO()
            with patch.object(negotiating_composer, "check_audio_ready") as ready:
                with patch.object(negotiating_composer, "build_llm_config", return_value=(False, "offline-model")):
                    with patch.object(NegotiatingComposerAgent, "generate_reply", return_value=response_json()) as generate:
                        with patch.object(negotiating_composer, "render_abc_to_wav", side_effect=self.fake_render) as render:
                            with redirect_stdout(stdout):
                                status = negotiating_composer.main([
                                    "--prompt", "A hopeful piano song.", "--output-dir", str(output),
                                ])
            self.assertEqual(status, 0)
            ready.assert_called_once()
            generate.assert_called_once()
            render.assert_called_once()
            for name in ("composition.abc", "composition.json", "composition.mid", "composition.wav"):
                self.assertTrue((output / name).is_file(), name)
            record = json.loads((output / "composition.json").read_text())
            self.assertEqual(record["model"], "offline-model")
            self.assertEqual((output / "composition.abc").read_text().strip(), record["response"]["abc"])
            self.assertIn("composition.wav", stdout.getvalue())

    def test_render_failure_preserves_previous_batch_and_saves_failed_score(self):
        with TemporaryDirectory(prefix="composer-cli-render-fail-") as directory:
            output = Path(directory)
            previous = {
                "composition.abc": b"previous score",
                "composition.json": b"previous record",
                "composition.mid": b"previous MIDI",
                "composition.wav": b"previous WAV",
            }
            for name, data in previous.items():
                (output / name).write_bytes(data)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(negotiating_composer, "check_audio_ready"):
                with patch.object(negotiating_composer, "build_llm_config", return_value=(False, "offline-model")):
                    with patch.object(NegotiatingComposerAgent, "generate_reply", return_value=response_json()) as generate:
                        with patch.object(negotiating_composer, "render_abc_to_wav", side_effect=RuntimeError("offline render failed")):
                            with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                                negotiating_composer.main([
                                    "--prompt", "A replacement piano song.", "--output-dir", str(output),
                                ])
            self.assertEqual(caught.exception.code, 1)
            generate.assert_called_once()
            for name, data in previous.items():
                self.assertEqual((output / name).read_bytes(), data, name)
            self.assertIn("X:1", (output / "failed_render.abc").read_text())
            self.assertNotIn("Saved ", stdout.getvalue())
            self.assertIn("render", stderr.getvalue().lower())

    def test_missing_audio_runtime_fails_before_requesting_music(self):
        with TemporaryDirectory(prefix="composer-no-audio-") as directory:
            stderr = io.StringIO()
            with patch.object(negotiating_composer, "check_audio_ready", side_effect=RuntimeError("Run setup_audio first")):
                with patch.object(NegotiatingComposerAgent, "generate_reply") as generate:
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                        negotiating_composer.main([
                            "--prompt", "A piano song.", "--output-dir", directory,
                        ])
            self.assertEqual(caught.exception.code, 1)
            generate.assert_not_called()
            self.assertIn("setup_audio", stderr.getvalue())
            self.assertFalse((Path(directory) / "composition.wav").exists())


if __name__ == "__main__":
    unittest.main()
