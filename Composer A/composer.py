"""One composer agent generates synchronized melody and harmony; no dependencies."""
import argparse
import array
import json
import io
import math
import os
from pathlib import Path
import shlex
import ssl
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from composition_results import reserve_result


def load_dotenv(paths=None):
    """Load supported settings without executing shell code or replacing exports."""
    if paths is None:
        script_dir = Path(__file__).resolve().parent
        paths = [Path.cwd() / '.env', script_dir / '.env', script_dir.parent / '.env']
    supported = {'OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_BASE_URL', 'OPENAI_ENDPOINT',
                 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT', 'AZURE_OPENAI_ENDPOINT'}
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if line.startswith('export '):
                line = line[7:].lstrip()
            name, separator, value = line.partition('=')
            name = name.strip()
            if not separator or name not in supported or name in os.environ:
                continue
            try:
                parts = shlex.split(value, comments=True, posix=True)
            except ValueError:
                raise ValueError(f'Invalid quoting in {path.name} at line {number}') from None
            os.environ[name] = ' '.join(parts)


SYSTEM_PROMPT = """You are one expert composer responsible for the ENTIRE composition:
melody, harmony, phrasing, instrumentation, and self-review. Do not delegate.
Compose melody and harmony together, with coherent motifs, varied rhythms, smooth
voice leading, chord tones on strong beats, and a convincing final cadence.
Follow the user's musical description. Return ONLY a JSON object with this schema:
{"title":"...","key":"C major","tempo":100,"melody_program":0,
"harmony_program":0,"bars":[{"chord":"C","melody":[{"pitches":[60],"ticks":4}],
"harmony":[{"pitches":[48,52,55],"ticks":16}]}]}
All music is 4/4. One tick is a sixteenth note. Each part in EACH bar must total
exactly 16 ticks. Melody events have one MIDI pitch or [] for rests. Harmony
events have 2-6 distinct simultaneous MIDI pitches or [] for rests. Pitches are
integers 21-108, ticks integers 1-16. Include audible melody AND harmony in every
bar. Choose actual sounding harmony notes, not just chord labels. Programs are
General MIDI integers 0-127. Tempo is an integer 40-240. Key is descriptive;
all MIDI pitches are absolute, including accidentals. Review all bars yourself
before responding. Use exactly the requested number of bars. No markdown."""


def validate_score(score, bars):
    if not isinstance(score, dict):
        raise ValueError('Composition must be a JSON object')
    for field in ('title', 'key'):
        value = score.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
            raise ValueError(f'{field} must be a nonempty single line (at most 200 characters)')
    for field, low, high in [('tempo', 40, 240), ('melody_program', 0, 127), ('harmony_program', 0, 127)]:
        if type(score.get(field)) is not int or not low <= score[field] <= high:
            raise ValueError(f'{field} must be an integer from {low} to {high}')
    if not isinstance(score.get('bars'), list) or len(score['bars']) != bars:
        raise ValueError(f'Expected exactly {bars} bars')
    for i, bar in enumerate(score['bars'], 1):
        if not isinstance(bar, dict) or not isinstance(bar.get('chord'), str) or not bar['chord'].strip() or len(bar['chord']) > 40 or any(c in bar['chord'] for c in '\n\r"\\'):
            raise ValueError(f'Bar {i}: invalid chord label')
        for part in ('melody', 'harmony'):
            events = bar.get(part)
            if not isinstance(events, list) or not events or len(events) > 16:
                raise ValueError(f'Bar {i}: {part} needs 1-16 events')
            total = 0
            audible = False
            for event in events:
                if not isinstance(event, dict) or type(event.get('ticks')) is not int or not 1 <= event['ticks'] <= 16:
                    raise ValueError(f'Bar {i}: invalid {part} duration')
                pitches = event.get('pitches')
                if not isinstance(pitches, list) or any(type(p) is not int or not 21 <= p <= 108 for p in pitches) or len(set(pitches)) != len(pitches):
                    raise ValueError(f'Bar {i}: invalid {part} pitches')
                if pitches and not (len(pitches) == 1 if part == 'melody' else 2 <= len(pitches) <= 6):
                    raise ValueError(f'Bar {i}: invalid {part} voicing')
                audible |= bool(pitches)
                total += event['ticks']
            if total != 16 or not audible:
                raise ValueError(f'Bar {i}: {part} must total 16 ticks and contain notes (got {total})')
    return score


def create_ssl_context():
    context = ssl.create_default_context()
    # python.org macOS builds can lack their own bundle even when macOS has one.
    # Preserve explicit certificate configuration and normal trust on other systems.
    if (sys.platform == 'darwin' and not context.get_ca_certs()
            and not os.environ.get('SSL_CERT_FILE') and not os.environ.get('SSL_CERT_DIR')
            and Path('/etc/ssl/cert.pem').is_file()):
        context.load_verify_locations(cafile='/etc/ssl/cert.pem')
    return context


def open_request(request):
    try:
        return urllib.request.urlopen(request, timeout=120, context=create_ssl_context())
    except urllib.error.URLError as exc:
        if sys.platform != 'darwin' or not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        # Apple's curl uses native certificate handling, including intermediate
        # discovery. Keep secrets on stdin, never in process arguments or logs.
        config = ['url = ' + json.dumps(request.full_url), 'request = ' + json.dumps(request.get_method())]
        if request.data is not None:
            config.append('data = ' + json.dumps(request.data.decode('utf-8')))
        config.extend('header = ' + json.dumps(f'{name}: {value}') for name, value in request.header_items())
        try:
            result = subprocess.run(['/usr/bin/curl', '--disable', '--silent', '--show-error',
                                     '--max-time', '120', '--write-out', '\n%{http_code}',
                                     '--config', '-'], input='\n'.join(config),
                                    capture_output=True, text=True, timeout=130)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError('macOS HTTPS request could not complete') from None
        if result.returncode:
            raise RuntimeError('macOS HTTPS connection failed; check network access and trusted certificates') from None
        body, _, status = result.stdout.rpartition('\n')
        if not status.isdigit():
            raise RuntimeError('macOS HTTPS client returned an invalid response')
        response = io.BytesIO(body.encode('utf-8'))
        if not 200 <= int(status) < 300:
            raise urllib.error.HTTPError(request.full_url, int(status), 'Request failed', {}, response)
        return response


def request_completion(messages, model, api_key, base_url):
    responses_api = base_url.rstrip('/').endswith('/openai/v1')
    body = ({'model': model.removeprefix('openai/'), 'input': messages,
             'text': {'format': {'type': 'json_object'}}, 'store': False, 'background': False}
            if responses_api else {'model': model, 'messages': messages,
                                   'response_format': {'type': 'json_object'}})
    payload = json.dumps(body).encode()
    route = '/responses' if responses_api else '/chat/completions'
    request = urllib.request.Request(base_url.rstrip('/') + route, data=payload,
                                    headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    try:
        with open_request(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            # Never echo the provider's message: it may include part of the key.
            raise RuntimeError('Authentication rejected (HTTP 401). Check that the key belongs to the configured endpoint. UNC course keys require OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu. Exported shell variables override .env.') from None
        raise RuntimeError(f'Model request failed (HTTP {exc.code}); check your key, model, and endpoint') from None
    except (urllib.error.URLError, TimeoutError) as exc:
        if isinstance(getattr(exc, 'reason', None), ssl.SSLCertVerificationError):
            raise RuntimeError('HTTPS certificate verification failed. Configure SSL_CERT_FILE with a trusted CA bundle for your Python installation or network.') from None
        raise RuntimeError(f'Model connection failed: {exc.reason if hasattr(exc, "reason") else "timeout"}') from None
    try:
        if responses_api:
            content = ''.join(part['text'] for item in result['output']
                              if item.get('type') == 'message'
                              for part in item.get('content', []) if part.get('type') == 'output_text')
        else:
            content = result['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise ValueError()
        return content
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError('Model endpoint returned no text completion') from None


def compose(prompt, bars=16, model='gpt-4o-mini', api_key=None,
            base_url='https://api.openai.com/v1', retries=2, completion=None):
    if not prompt.strip() or not 1 <= bars <= 64 or not 0 <= retries <= 5:
        raise ValueError('Use a nonempty prompt, 1-64 bars, and 0-5 repair retries')
    if completion is None:
        if not api_key:
            raise ValueError('Set OPENAI_API_KEY for model generation, or use --demo for an offline smoke test')
        completion = lambda messages: request_completion(messages, model, api_key, base_url)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Compose exactly {bars} bars. Request: {prompt}'}]
    for attempt in range(retries + 1):
        content = completion(messages)
        try:
            return validate_score(json.loads(content), bars), messages + [{'role': 'assistant', 'content': content}]
        except (ValueError, TypeError) as exc:
            if attempt == retries:
                raise ValueError(f'Composer returned invalid music after {attempt + 1} attempts: {exc}') from None
            messages.extend([{'role': 'assistant', 'content': content},
                             {'role': 'user', 'content': f'Validation failed: {exc}. Return the complete corrected JSON composition.'}])


def demo_score(bars=16):
    """Fixed musical fixture, NOT an AI or prompt-conditioned composition."""
    chords = [('C', [48, 52, 55], [72, 74, 76, 79]),
              ('Am', [45, 52, 57], [76, 72, 69, 72]),
              ('F', [41, 48, 53], [69, 72, 77, 76]),
              ('G', [43, 50, 55], [74, 71, 67, 71])]
    score = dict(title='Together in C', key='C major', tempo=108,
                 melody_program=0, harmony_program=0, bars=[])
    for i in range(bars):
        label, harmony, melody = chords[i % 4]
        if i == bars - 1:
            label, harmony, melody = 'C', [48, 52, 55], [72]
        durations = [16] if len(melody) == 1 else [4, 2, 2, 8]
        score['bars'].append(dict(chord=label,
            melody=[dict(pitches=[p], ticks=d) for p, d in zip(melody, durations)],
            harmony=[dict(pitches=harmony, ticks=16)]))
    return validate_score(score, bars)


def abc_pitch(pitch):
    # Explicit naturals prevent accidental carry within a measure; K:C is intentional.
    names = ['=C', '^C', '=D', '^D', '=E', '=F', '^F', '=G', '^G', '=A', '^A', '=B']
    name = names[pitch % 12]
    octave = pitch // 12 - 1
    return name + ("," * (4 - octave) if octave < 4 else "'" * (octave - 4))


def to_abc(score):
    lines = ['X:1', 'T:' + score['title'], 'C:Composer A', 'M:4/4', 'L:1/16',
             f'Q:1/4={score["tempo"]}', '%%score (1 2)',
             'V:1 name="Melody" clef=treble', 'V:2 name="Harmony" clef=bass',
             'K:C', '% Sounding pitches are explicit; intended key: ' + score['key']]
    for voice, part in enumerate(('melody', 'harmony'), 1):
        lines.extend([f'V:{voice}', f'%%MIDI program {score[part + "_program"]}'])
        measures = []
        for bar in score['bars']:
            notes = []
            for event in bar[part]:
                pitches = event['pitches']
                token = 'z' if not pitches else abc_pitch(pitches[0]) if len(pitches) == 1 else '[' + ''.join(map(abc_pitch, pitches)) + ']'
                notes.append(token + str(event['ticks']))
            measures.append(('"' + bar['chord'] + '"' if part == 'melody' else '') + ' '.join(notes))
        lines.append(' | '.join(measures) + ' |]')
    return '\n'.join(lines) + '\n'


def vlq(value):
    data = [value & 127]
    while value > 127:
        value >>= 7
        data.insert(0, (value & 127) | 128)
    return bytes(data)


def to_midi(score):
    tracks = []
    for channel, part in enumerate(('melody', 'harmony')):
        events = []
        if channel == 0:
            events.append((0, b'\xff\x51\x03' + int(60000000 / score['tempo']).to_bytes(3, 'big')))
            events.append((0, b'\xff\x58\x04\x04\x02\x18\x08'))
        events.append((0, bytes([0xC0 | channel, score[part + '_program']])))
        tick = 0
        for bar in score['bars']:
            for event in bar[part]:
                for pitch in event['pitches']:
                    events.append((tick, bytes([0x90 | channel, pitch, 90 if channel == 0 else 60])))
                    events.append((tick + event['ticks'] * 120, bytes([0x80 | channel, pitch, 0])))
                tick += event['ticks'] * 120
        events.sort(key=lambda item: item[0])  # stable: old note-offs precede next note-ons
        data, previous = bytearray(), 0
        for when, message in events:
            data.extend(vlq(when - previous) + message)
            previous = when
        data.extend(vlq(tick - previous) + b'\xff\x2f\x00')
        tracks.append(b'MTrk' + struct.pack('>I', len(data)) + data)
    return b'MThd' + struct.pack('>IHHH', 6, 1, 2, 480) + b''.join(tracks)


def write_wav(score, path, sample_rate=16000):
    # Stream one bar at a time; quiet additive synth is a portable preview.
    samples_per_tick = sample_rate * 60 / score['tempo'] / 4
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
        for bar_index, bar in enumerate(score['bars']):
            bar_start = round(bar_index * 16 * samples_per_tick)
            size = round((bar_index + 1) * 16 * samples_per_tick) - bar_start
            samples = [0.0] * size
            for part in ('melody', 'harmony'):
                tick = bar_index * 16
                for event in bar[part]:
                    start = round(tick * samples_per_tick) - bar_start
                    end = round((tick + event['ticks']) * samples_per_tick) - bar_start
                    length = end - start
                    gain = .22 if part == 'melody' else .16 / max(1, len(event['pitches']))
                    for pitch in event['pitches']:
                        omega = math.tau * 440 * 2 ** ((pitch - 69) / 12) / sample_rate
                        for n in range(length):
                            envelope = min(1, n / (sample_rate * .01), (length - 1 - n) / (sample_rate * .03))
                            samples[start + n] += gain * envelope * math.sin(omega * n)
                    tick += event['ticks']
            pcm = array.array('h', (round(max(-1, min(1, sample)) * 32767) for sample in samples))
            if sys.byteorder != 'little':
                pcm.byteswap()
            output.writeframes(pcm.tobytes())


def export(score, output_dir, metadata):
    validate_score(score, len(score['bars']))
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'composition.json').write_text(json.dumps(score, indent=2) + '\n', encoding='utf-8')
    (folder / 'composition.abc').write_text(to_abc(score), encoding='utf-8')
    (folder / 'composition.mid').write_bytes(to_midi(score))
    write_wav(score, folder / 'composition.wav')
    (folder / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description='One agent composes both melody and harmony in 4/4.')
    try:
        load_dotenv()
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error loading .env: {exc}\n')
    parser.add_argument('-p', '--prompt', required=True)
    parser.add_argument('-o', '--output-dir', '--output_dir', help='Parent folder for automatically numbered A results')
    parser.add_argument('--bars', type=int, default=16, choices=range(1, 65), metavar='1-64')
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', os.environ.get('OPENAI_ENDPOINT', '')).strip()
    default_base = endpoint.rstrip('/') + '/openai/v1' if endpoint else os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    parser.add_argument('--model', default=os.environ.get('AZURE_OPENAI_DEPLOYMENT', os.environ.get('OPENAI_MODEL', '' if endpoint else 'gpt-4o-mini')))
    parser.add_argument('--base-url', default=default_base)
    parser.add_argument('--retries', type=int, default=2, choices=range(6))
    parser.add_argument('--demo', action='store_true', help='Use fixed offline test music; does not interpret the prompt')
    args = parser.parse_args(argv)
    try:
        if not args.prompt.strip():
            raise ValueError('Prompt cannot be empty')
        if args.demo:
            score, messages = demo_score(args.bars), []
        else:
            if not args.model.strip():
                raise ValueError('Set OPENAI_MODEL or AZURE_OPENAI_DEPLOYMENT to your enabled course deployment')
            api_key = os.environ.get('AZURE_OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY'))
            score, messages = compose(args.prompt, args.bars, args.model, api_key, args.base_url, args.retries)
        folder = export(score, reserve_result('A', args.output_dir), dict(prompt=args.prompt, mode='offline-demo' if args.demo else 'single-agent',
                        model=None if args.demo else args.model, messages=messages))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Created {len(score["bars"])} bars of melody and harmony in {folder.resolve()}')
    return 0


if __name__ == '__main__':
    main()
