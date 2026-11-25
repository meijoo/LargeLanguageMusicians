# Negotiating Composer

To negotiate a song with the existing `ComposerSingle` agent, run from the
repository root in the composer environment:

```bash
python -m negotiating_composer.negotiate \
  --prompt "write a jazz and blues inspired piano song" \
  --output-dir "negotiated song"
```

The controller obtains independent drafts, then alternates up to six global
negotiation turns. It requires two consecutive acceptances of the identical
plan before asking ComposerSingle to implement the final song. It preserves
each composer's existing composition format and converts ComposerSingle's
validated score to ABC for peer review. The controller itself makes no musical
decisions and does not claim agreement when the agents disagree.

The output folder contains `negotiation.md`, full structured exchanges in
`negotiation.json`, each composer's conversation history, the initial
ComposerSingle draft, and the final `composition.json`, `.abc`, `.mid`, and
`.wav`. Audio uses ComposerSingle's built-in synthesizer. Logs are checkpointed
after every successful exchange; API failures leave partial logs rather than a
fabricated song. Use a fresh folder for another run after any exchanges have
been recorded. Failed attempts with no exchanges can be retried in place.

`NegotiatingComposerAgent` in [agent.py](agent.py) is **one agent** that composes
melody, harmony, rhythm, form, and instrumentation
together. It extends the AutoGen `AssistantAgent` used by ComposerX and does not
create specialist agents or start a group chat. It can compose independently now
and exchange proposals with another complete composer later.

The agent code, tests, dependencies, and documentation live in
`negotiating_composer/`. The original `ComposerX/` source and README are unmodified.
The configuration, environment,
rendering cache, and generated music remain at the repository root so the
existing setup continues to work.

Use Python 3.11 and a separate environment for this extension. Its dependencies
are separate because the original scripts use the legacy OpenAI 0.28 interface.
From the repository root:

```bash
python3.11 -m venv .venv-composer
source .venv-composer/bin/activate
python -m pip install -r negotiating_composer/requirements.txt
python -m negotiating_composer.setup_audio
cp -n negotiating_composer/.env.example .env
```

Edit `.env` in the repository root and enter your UNC Azure subscription key
and exact enabled deployment alias:

```dotenv
OPENAI_API_KEY=your-UNC-subscription-key
OPENAI_MODEL=gpt-5.6-luna
OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu
```

`gpt-5.6-luna` is the LearnWithAI example alias; confirm that it is enabled for
your subscription. Clear a previously exported key with `unset OPENAI_API_KEY`
if you want the key in `.env` to take effect. Then run:

```bash
python -m negotiating_composer \
  --prompt "Write a hopeful 16-bar piano piece in 4/4." \
  --preference "Use a lyrical, mostly stepwise melody." \
  --preference "Use sustained accompaniment in the opening." \
  --peer-preference "Use syncopation in the middle section." \
  --peer-preference "Include a contrasting bridge." \
  --output-dir results/composer_a
```

The agent loads the repository-root `.env` when constructed; existing shell
variables override values with the same name. Azure aliases take precedence:
`AZURE_OPENAI_API_KEY` over `OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` over
`OPENAI_MODEL`, and `AZURE_OPENAI_ENDPOINT` over `OPENAI_ENDPOINT`. Configure
one naming scheme to avoid stale overrides. An explicit `--model` or constructor
`model` argument overrides the environment's deployment/model choice.

For `OPENAI_ENDPOINT`/`AZURE_OPENAI_ENDPOINT`, supply the origin; the client
appends `/openai/v1/` once and uses the **Responses API** through a small AutoGen
model-client adapter. It sends the subscription key in Azure's `api-key` header
and standard bearer authentication. This follows the
[Azure v1 client contract](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle).
No legacy `api-version` setting is needed. The exact UNC deployment name is
preserved, including dots; there is no `azure/` provider prefix. Azure mode
requires a deployment in the environment or an explicit model argument.

Public OpenAI and existing compatible Chat Completions configurations still
work when the endpoint aliases are absent. `OPENAI_BASE_URL` selects a complete
base URL for that legacy configuration path; it is ignored when an endpoint
alias is set. The original `gpt-6.1-luna` project default remains a fallback
only for public/compatible mode, so Azure never silently selects an unknown
deployment. No credentials are loaded from `OAI_CONFIG_LIST` unless supplied
explicitly through an AutoGen `llm_config`. `.env` is excluded from Git, and
generated artifacts contain the resolved model name without credentials.

The command makes one composition request and automatically saves:

- `composition.abc`: a complete instrumental score with sounding melody and
  accompaniment, ready for the existing ABC/MIDI/audio tooling.
- `composition.json`: the rationale, implemented plan, preferences, and message
  history, kept separate from the score.
- `composition.mid`: the complete score as MIDI, with the written voice instruments.
- `composition.wav`: a playable stereo piano/instrument recording.
- `composition.render.json`: rendering settings, source and SoundFont hashes,
  and duration for reproducibility.

Play the WAV on macOS with:

```bash
afplay results/composer_a/composition.wav
```

Audio setup is needed only once per checkout/Python environment; it is already
installed in this workspace. It puts the synthesizer and TimGM6mb General MIDI
SoundFont in the ignored `.composerx-audio` directory. The SoundFont's license
and pinned source information are stored alongside it. Rendering then works
locally without network access, MuseScore, `abc2midi`, or system audio libraries.
The same generation command always creates WAV; no extra rendering flag is
needed. Missing audio setup is detected before a model request is made.

Each written voice is rendered with its MIDI program on a separate channel.
Repeated voice declarations are merged in memory and chord-name annotations
are not turned into extra accompaniment; the original ABC remains unchanged.
The WAV uses 44.1 kHz, 16-bit stereo audio with a two-second release tail.
Existing output files are replaced only after audio rendering succeeds. A
rendering failure saves the new draft as `failed_render.abc` for diagnosis and
preserves the previous successful files.

To render an existing ABC without generating another composition:

```bash
python -m negotiating_composer.render_audio results/composer_a/composition.abc
```

To connect this one agent to a future negotiating partner, use its Python API:

```python
from pathlib import Path
from negotiating_composer.agent import NegotiatingComposerAgent

composer = NegotiatingComposerAgent(
    name="ComposerA",
    preferences=["Stepwise melody", "Sustained opening accompaniment"],
    peer_preferences=["Syncopated middle section", "Contrasting bridge"],
)
draft = composer.compose("Write a hopeful 16-bar piano piece in 4/4.")
Path("draft.abc").write_text(draft.abc, encoding="utf-8")

# Later, a controller obtains this message from the other composer.
peer_message = "Keep the stepwise melody; add syncopation in bars 9-12."
reply = composer.respond(peer_message, turn=2)
print(reply.to_json())  # Forward this structured message to the peer.

# Only after the controller has confirmed agreement from BOTH composers:
agreed_plan = [
    "Use a stepwise melody in all 16 bars.",
    "Sustain accompaniment in bars 1-8; use syncopation in bars 9-12.",
    "Contrast bars 9-12, then return to the opening motif in bars 13-16.",
]
final = composer.finalize(agreed_plan)
Path("final.abc").write_text(final.abc, encoding="utf-8")
```

`compose(prompt)` starts a fresh session and makes one model request. Calling it
again clears prior musical context only after successful generation. For the
independent condition, use only `compose`; both preference sets can be supplied
in both conditions to keep the musical brief identical.

`respond(peer_message, turn=...)` accepts plain text or another
`ComposerResponse`. Supply the **global** negotiation turn (1-6), increasing
across calls; for example, this composer might speak on turns 2, 4, and 6 while
its partner speaks on 1, 3, and 5. The future controller owns that shared counter
and routes messages. This agent checks its supplied turn numbers but cannot
verify the partner's counter. Initial independent drafts and final rendering
are outside these six negotiation turns. The same model composes and discusses
the music. No extra reviewer or manager model is created.

Each response contains `message`, `plan`, `abc`, and `accepts_peer_plan`. The
acceptance flag represents this agent's acceptance of the peer's latest exact
plan; it does not establish mutual agreement or stop negotiation automatically.
`finalize(agreed_plan)` requires an explicit plan confirmed by the caller, makes
one rendering request, and locks that session. Start a new piece with `compose`.
The requested agreement is retained in history for later implementation checks.
Use these methods consistently for a session; inherited AutoGen chat methods
have separate history and local automatic-reply counters.

Every generation requests a strict JSON Schema for the four response fields.
The Responses client sends it through `text.format`; the Chat Completions path
uses `response_format`. This uses
[Azure structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)
instead of relying only on prompt instructions. The selected deployment must
support structured outputs. There is no automatic retry or repair composition
when a reply is malformed.

Response parsing still checks JSON types, required ABC headers, and the presence of
instrumented, sounding voices. These are structural checks; they do not certify
ABC syntax, bar durations, musical quality, preference satisfaction, or agreement
implementation. Use an ABC renderer and the project's listener/adherence ratings
for those checks. API/response errors propagate without consuming a negotiation
turn or discarding a previous successful composition. If the CLI receives an
invalid model reply, it saves the raw text to `failed_response.txt` in the output
directory and reports which JSON fields are missing or unexpected. Existing
`composition.abc` and `composition.json` files are preserved on that failure.
The extension does not run the two-agent experiment or listener evaluation itself.

Run the offline tests in the new environment (model replies are mocked):

```bash
python -m unittest discover -s negotiating_composer/tests -v
```
