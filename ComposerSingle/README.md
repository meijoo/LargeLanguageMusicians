# ComposerSingle

A standalone single-agent adaptation of the neighboring ComposerX project.
One composer receives your description and composes **both melody and playable
harmony together**, including chord progression, tempo, and instrumentation.
There is no leader, group chat, separate harmony/melody agent, or reviewer agent.
ComposerX is left unchanged.

## Run

### UNC course credentials

The COMP 590 repositories use UNC subscription keys, which are not public OpenAI
API keys. Set the following in the repository-root `.env`, keeping your existing key:

```dotenv
OPENAI_API_KEY=your-unc-subscription-key
OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu
OPENAI_MODEL=gpt-5-mini
```

Use an enabled deployment for your subscription. UNC mode appends `/openai/v1/`
to the endpoint origin and uses the Responses API with bearer authentication,
`store=False`, and `background=False`, matching `590_agents/provider.py`.
The Azure aliases `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, and
`AZURE_OPENAI_ENDPOINT` take precedence over the corresponding `OPENAI_` names.
`--model` overrides the configured deployment; an `openai/` prefix is removed.
No Docker is required. Docker in `agent_evals` runs the evaluation sandboxes.

### Public OpenAI credentials

Requires Python 3.10+ and an OpenAI API key. There are no Python package dependencies
and no MuseScore, abc2midi, AutoGen, or soundfont requirement.

From the repository root:

Save `OPENAI_API_KEY=your-api-key` in `.env` in the repository root or
`ComposerSingle/`. The script loads it automatically; no `export` is needed.
Quoted values, `export KEY=value`, and inline comments are supported.
Existing environment variables take priority, followed by `.env` in the current
directory, `ComposerSingle/`, and the repository root. Model and base URL settings
can also go in `.env`. Values are read as text, never executed as shell commands.
Alternatively, export the key in your terminal:

```bash
export OPENAI_API_KEY='your-api-key'
python3 ComposerSingle/composer.py \
  -p 'Compose a gentle piano piece in C major with a lyrical melody and warm chords' \
  -o ComposerSingle/output/piano
```

The default is 16 bars in 4/4 and model `gpt-4o-mini`. Use `--bars 8`,
`--model YOUR_MODEL`, and `--retries 2` to customize. `OPENAI_MODEL` and
`OPENAI_BASE_URL` also set defaults. `--base-url` must identify a trusted
OpenAI-compatible API base ending in `/v1` that supports Chat Completions and
JSON object responses; your API key is sent to this endpoint. When `OPENAI_ENDPOINT`
or its Azure alias is set, that origin selects UNC Responses mode instead of the
default Chat Completions configuration. Explicit `--base-url` takes priority.
The existing
ComposerX `OAI_CONFIG_LIST` is not automatically loaded. Live requests incur your
provider's usual API charges.

The same agent returns a structured score with actual notes in both parts.
Local validation checks pitch ranges, voicing, bar count, and exact rhythmic
alignment. If the result is invalid, feedback goes back to the **same agent in
the same conversation**, up to the repair limit. Network and authentication
errors terminate clearly. There is no unbounded restart loop.

## Output

Each run writes five files to the chosen directory (existing files with these
names are replaced):

- `composition.json`: title, intended key, tempo, MIDI programs, and both parts.
- `composition.abc`: a two-voice score with chord labels.
- `composition.mid`: synchronized melody and harmony on separate MIDI tracks.
- `composition.wav`: a portable synthesized audio preview of both parts.
- `run.json`: original request, model/mode, and the agent conversation; no API key.

The WAV preview uses simple sine tones rather than the selected instrument
sounds. MIDI preserves the chosen General MIDI programs for playback in a DAW
or notation program. ABC uses `K:C` with explicit accidentals for every sounding
pitch; the intended key is preserved in a comment and the JSON. Durations are
integer sixteenth-note ticks, with exactly 16 ticks in each part of each 4/4 bar.
Supported lengths are 1-64 bars. This version does not support other meters,
triplets, or arbitrary instrument parts. Validation checks structural correctness;
musical taste and adherence to a style still depend on the model.

## Offline example and tests

```bash
python3 ComposerSingle/composer.py \
  -p 'Offline pipeline test' -o ComposerSingle/examples/demo --demo
python3 -m unittest discover -s ComposerSingle/tests -v
```

`--demo` uses a fixed C-major fixture. It does **not** interpret the prompt or call
an AI model. It allows testing and listening without credentials. The included
16-bar example was generated using this mode.

Fifteen automated tests cover UNC request routing and configuration aliases, native macOS HTTPS fallback,
authentication error redaction, certificate verification and the macOS CA fallback,
.env loading and precedence, CLI configuration,
a single joint model response, bounded repairs within
the same conversation, malformed music rejection, mocked HTTP request/response,
ABC/MIDI/WAV exports (including MIDI note pairing and duration), CLI execution
from another directory, and missing credentials. Tests use synthetic model
responses; live model generation is not part of the automated tests.
An end-to-end live run with the UNC `gpt-5-mini` deployment successfully generated
a validated 16-bar melody and harmony, plus ABC, MIDI, WAV, and run metadata.

On macOS, if Python has no default trusted certificates, the script loads the
system CA bundle at `/etc/ssl/cert.pem` while keeping certificate and hostname
verification enabled. Explicit `SSL_CERT_FILE` or `SSL_CERT_DIR` settings take
priority. HTTPS connectivity to OpenAI was verified without sending an API key.
If Python cannot verify a chain on macOS, the script retries using Apple's
`/usr/bin/curl` with native certificate verification. Secrets are passed through
stdin rather than process arguments. It never disables certificate verification.

## Relationship to ComposerX

This implementation follows ComposerX's text-to-symbolic-music approach, ABC
output, separate melody/harmony voices, and musical guidance on motifs, phrasing,
and harmony. It replaces the multi-agent orchestration with one system prompt
and replaces external conversion tools with local exporters. It does not import
the original experimental single-agent scripts, whose import-time batch jobs
reference missing datasets and configuration files.
