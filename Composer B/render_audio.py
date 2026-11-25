"""Render ComposerX ABC scores to MIDI and WAV without external music apps."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Optional, Union
import wave

if __package__:
    from .audio_runtime import check_audio_ready, load_synth_module
else:
    from audio_runtime import check_audio_ready, load_synth_module


SAMPLE_RATE = 44100
RELEASE_SECONDS = 2


def _prepare_voices(abc: str) -> tuple[str, list[tuple[int, Optional[int]]]]:
    """Merge repeated V: declarations in memory; retain each voice's program."""
    # An inline voice switch and a separate V: line have the same meaning here.
    abc = re.sub(r"\[V:([^\]]+)\]", r"\nV:\1\n", abc)
    headers, voices = [], {}
    current = None
    header_open = True
    for line in abc.splitlines():
        match = re.match(r"^\s*V:\s*(\S+)", line)
        if match:
            current = match.group(1)
            voices.setdefault(current, {"declaration": line.strip(), "lines": []})
        elif header_open:
            # V: definitions are legal before the first K:, but the initial
            # key/meter/tempo still belong to the whole tune. Keep them ahead
            # of every voice body when regrouping declarations.
            if current is not None and line.lstrip().startswith("%%MIDI"):
                voices[current]["lines"].append(line)
            else:
                headers.append(line)
            if re.match(r"^\s*K:", line):
                header_open = False
        elif current is None:
            headers.append(line)
        else:
            voices[current]["lines"].append(line)
    programs = []
    for voice in voices.values():
        body = "\n".join(voice["lines"])
        directives = re.findall(r"^%%MIDI program[ \t]+(?:(\d+)[ \t]+)?(\d+)[ \t]*$", body, re.MULTILINE)
        if len({program for _, program in directives}) > 1:
            raise ValueError("WAV rendering requires one MIDI program per voice.")
        channel, program = directives[0] if directives else ("", "0")
        channel_directive = re.search(r"^%%MIDI channel[ \t]+(\d+)[ \t]*$", body, re.MULTILINE)
        if channel_directive:
            channel = channel_directive.group(1)
        if not 0 <= int(program) <= 127 or (channel and not 1 <= int(channel) <= 16):
            raise ValueError("Invalid MIDI program or channel in ABC score.")
        programs.append((int(program), int(channel) - 1 if channel else None))
    normalized = "\n".join(headers + [
        line for voice in voices.values() for line in [voice["declaration"], *voice["lines"]]
    ])
    return normalized, programs


def render_abc_to_wav(
    abc_path: Union[str, Path], output_dir: Optional[Path] = None, *,
    soundfont: Optional[Path] = None,
) -> Path:
    """Save sibling MIDI, stereo PCM WAV, and rendering metadata from a score.

    The original ABC is preserved. Rendering uses no API requests or runtime
    downloads, and output files are published only after synthesis succeeds.
    """
    font_path = check_audio_ready(soundfont)
    import numpy as np
    from music21 import converter, harmony, instrument, midi

    tinysoundfont = load_synth_module()
    source = Path(abc_path)
    source_bytes = source.read_bytes()
    normalized, programs = _prepare_voices(source_bytes.decode("utf-8"))
    if not normalized.strip():
        raise ValueError("Cannot render an empty ABC score.")
    score = converter.parseData(normalized, format="abc")
    # Chord names label harmony. Rendering them as extra notes would add an
    # accompaniment that the composer never actually wrote.
    for label in list(score.recurse().getElementsByClass(harmony.Harmony)):
        label.activeSite.remove(label)
    parts = list(score.parts)
    if not parts or any(not list(part.recurse().notes) for part in parts):
        raise ValueError("Cannot render a score with empty or silent voices.")
    if programs and len(parts) != len(programs):
        raise ValueError("ABC voices could not be mapped to MIDI parts reliably.")
    if len({part.duration.quarterLength for part in parts}) != 1:
        raise ValueError("ABC voices must have matching durations for WAV rendering.")
    reserved_channels = {channel for _, channel in programs if channel is not None}
    available_channels = [channel for channel in range(16) if channel != 9 and channel not in reserved_channels]
    assigned_channels = []
    for index, part in enumerate(parts):
        program, channel = programs[index] if programs else (0, None)
        for existing in list(part.recurse().getElementsByClass(instrument.Instrument)):
            existing.activeSite.remove(existing)
        selected = instrument.instrumentFromMidiProgram(program)
        if channel is None or channel in assigned_channels:
            if not available_channels:
                raise ValueError("Too many voices for distinct MIDI channels.")
            channel = available_channels.pop(0)
        # Independent channels prevent a note-off in one piano voice from
        # cutting off an overlapping note of the same pitch in another voice.
        selected.midiChannel = channel
        assigned_channels.append(channel)
        part.insert(0, selected)

    destination = Path(output_dir) if output_dir is not None else source.parent
    destination.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    with TemporaryDirectory(prefix=".audio-", dir=destination) as temporary:
        staging = Path(temporary)
        midi_path = staging / f"{stem}.mid"
        score.write("midi", fp=midi_path)
        # music21 reuses channels for equal instrument programs even when parts
        # request distinct channels. Reapply assignments to the finished tracks.
        midi_file = midi.MidiFile()
        midi_file.readstr(midi_path.read_bytes())
        note_tracks = [track for track in midi_file.tracks
                       if any(event.isNoteOn() for event in track.events)]
        if len(note_tracks) != len(assigned_channels):
            raise ValueError("MIDI tracks could not be mapped to the written voices.")
        for track, channel in zip(note_tracks, assigned_channels):
            track.setChannel(channel + 1)  # music21 numbers channels from 1.
        midi_path.write_bytes(midi_file.writestr())
        synth = tinysoundfont.Synth(gain=-6, samplerate=SAMPLE_RATE)
        sfid = synth.sfload(str(font_path))
        for channel in range(16):
            synth.program_select(channel, sfid, 0, 0)
        sequencer = tinysoundfont.Sequencer(synth)
        sequencer.midi_load(str(midi_path))
        if not sequencer.events:
            raise ValueError("The score produced no MIDI events.")
        last_event = max(event.t for event in sequencer.events)
        # The sequencer has no event for a trailing rest. Use the complete
        # tempo-aware score length as a floor so written silence is preserved.
        score_seconds = float(score.flatten().seconds)
        if not math.isfinite(score_seconds) or score_seconds <= 0:
            raise ValueError("Could not determine the score's playback duration.")
        total_frames = int((max(last_event, score_seconds) + RELEASE_SECONDS) * SAMPLE_RATE)
        chunks = []
        position = 0
        while position < total_frames:
            frames = min(4096, total_frames - position)
            samples = np.frombuffer(synth.generate(frames), dtype=np.float32).copy()
            chunks.append(samples)
            position += frames
        audio = np.concatenate(chunks)
        if not np.all(np.isfinite(audio)):
            raise ValueError("The synthesizer produced invalid audio samples.")
        peak = float(np.max(np.abs(audio)))
        if peak <= 1e-6:
            raise ValueError("The rendered audio is silent.")
        # Apply one scale to the whole piece if needed to avoid clipping.
        scale = min(1.0, 0.98 / peak)
        pcm = np.rint(np.clip(audio * scale, -1, 1) * 32767).astype("<i2")
        wav_path = staging / f"{stem}.wav"
        with wave.open(str(wav_path), "wb") as output:
            output.setparams((2, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            output.writeframes(pcm.tobytes())
        metadata = {
            "renderer": "TinySoundFont 0.3.7", "abc_converter": "music21 9.3.0",
            "soundfont": font_path.name,
            "soundfont_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
            "source_abc_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "sample_rate": SAMPLE_RATE, "channels": 2, "pcm_bits": 16,
            "duration_seconds": total_frames / SAMPLE_RATE,
            "gain_db": -6, "clipping_scale": scale,
            "voice_midi_programs": [program for program, _ in programs],
            "voice_midi_channels": [channel + 1 for channel in assigned_channels],
            "voice_note_counts": [len(part.recurse().notes) for part in parts],
            "chord_labels": "Annotations only; no added accompaniment.",
        }
        (staging / f"{stem}.render.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for artifact in staging.iterdir():
            artifact.replace(destination / artifact.name)
    return destination / f"{stem}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an existing ABC score to MIDI and WAV.")
    parser.add_argument("abc_path", type=Path)
    args = parser.parse_args()
    try:
        output = render_abc_to_wav(args.abc_path)
    except Exception as error:
        parser.exit(1, f"Audio rendering failed: {error}\n")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
