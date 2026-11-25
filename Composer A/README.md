# Composer A

Composer A is the independent composer formerly named ComposerSingle. One model
writes melody and playable harmony together as a validated JSON score. It uses
4/4, integer sixteenth-note durations, and 1–64 bars (16 by default). Invalid
scores can be repaired by the same model up to the configured retry limit.

Run from the repository root with the existing environment and root `.env`:

```bash
source .venv-composer/bin/activate
python -m composer_a --prompt "Write a jazz and blues inspired piano song."
```

Every run saves into a new `Single Composition Results/A_#` folder. A and B have
independent counters. Existing folders are never reused, and deleted numbers
below the current maximum are not filled. `-o PATH` selects a different parent
folder; the numbered A folder is still created inside it.

Each result contains `composition.json`, `.abc`, `.mid`, `.wav`, and `run.json`.
The WAV is a sine-tone preview; the MIDI preserves the chosen instruments.
`--bars`, `--model`, and `--retries` retain their existing meanings. `--demo`
produces a fixed offline fixture rather than interpreting the prompt.

```bash
python -m composer_a --prompt "Offline test" --demo -o /tmp/composer-check
python -m unittest discover -s "Composer A/tests" -v
```

UNC settings are read from `.env`: `OPENAI_API_KEY`, `OPENAI_MODEL`, and
`OPENAI_ENDPOINT=https://azureaiapi.cloud.unc.edu`. Azure-named aliases take
precedence. Shell variables override file values. Composer A has no third-party
Python dependencies. The original ComposerX directory is unchanged.
