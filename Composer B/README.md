# Composer B

Composer B is the complete composer formerly named negotiating_composer. It
composes all musical roles together and can critique, revise, and negotiate with
Composer A. Its AutoGen agent returns a complete ABC score, rationale, implemented
plan, and an explicit acceptance flag. It retains conversation history and uses
strict structured output. No specialist melody or harmony agents are created.

Run from the repository root:

```bash
source .venv-composer/bin/activate
python -m composer_b --prompt "Write a jazz and blues inspired piano song."
```

Each independent run creates `Single Composition Results/B_#` with a fresh number.
It saves ABC, JSON, MIDI, WAV, and rendering metadata. The WAV uses local General
MIDI SoundFont instruments. Each voice keeps its instrument, pitches, timing,
and rests. No API call is made for audio rendering.

To negotiate a song between Composer A and Composer B:

```bash
python -m composer_b.negotiate --prompt "Write a jazz and blues inspired piano song."
```

Each session creates `Negotiated Composition Results/AB_#`. Independent starting
drafts and negotiation logs remain inside that session folder. A speaks on odd
turns, B on even turns, up to six global negotiation turns. Mutual agreement
requires consecutive acceptance flags and identical plans. Composer A produces
the final song after agreement; this controller currently uses A's sine-tone
WAV exporter. `--full-six-turns` keeps all six turns even after early agreement.
Failed sessions retain their logs and allocated number, preserving every attempt.

`-o PATH` overrides the result parent for either command; a new `B_#` or `AB_#`
child is still created automatically. A and B counts are independent. Never reuse
an older folder to start a new session.

For a new installation:

```bash
python -m pip install -r "Composer B/requirements.txt"
python -m composer_b.setup_audio
cp -n "Composer B/.env.example" .env
```

The root `.env`, existing virtual environment, and `.composerx-audio` cache retain
their existing locations. Configure your UNC key, exact deployment alias in
`OPENAI_MODEL`, and `OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu`. Azure
aliases take precedence, shell values override `.env`, and `--model` overrides
the configured model for an independent B run.

Python imports use the small root launchers to support folder names with spaces:

```python
from composer_b.agent import NegotiatingComposerAgent
agent = NegotiatingComposerAgent(name="Composer B")
draft = agent.compose("A hopeful piano piece")
# Later: agent.respond(peer_message, turn=2), agent.finalize(agreed_plan)
```

```bash
python -m unittest discover -s "Composer B/tests" -v
```

The original ComposerX directory is unchanged. Historical result logs are retained
verbatim when moved, so they can contain the former composer names and paths.
