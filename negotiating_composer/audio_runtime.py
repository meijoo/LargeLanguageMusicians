"""Locate the local, offline audio renderer without loading API credentials."""

import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".composerx-audio"
RUNTIME_PACKAGES = RUNTIME_ROOT / "packages"
DEFAULT_SOUNDFONT = RUNTIME_ROOT / "TimGM6mb.sf2"
SOUNDFONT_SHA256 = "c5378b62028c920cb11e4803327983fee2f2cdff5dc89c708e39da417e51c854"
SETUP_COMMAND = "python -m negotiating_composer.setup_audio"


def _setup_error(detail: str) -> RuntimeError:
    return RuntimeError(f"Audio rendering is not ready: {detail}. Run {SETUP_COMMAND} "
                        "using the same Python environment, then retry.")


def load_synth_module() -> ModuleType:
    """Import the local TinySoundFont wheel; PyAudio is not used for rendering."""
    local_packages = str(RUNTIME_PACKAGES)
    if RUNTIME_PACKAGES.is_dir() and local_packages not in sys.path:
        sys.path.insert(0, local_packages)
        importlib.invalidate_caches()
    try:
        module = importlib.import_module("tinysoundfont")
        if not hasattr(module, "Synth") or not hasattr(module, "Sequencer"):
            raise ImportError("Incomplete TinySoundFont installation")
        return module
    except (ImportError, OSError) as error:
        raise _setup_error("the offline synthesizer is missing or incompatible") from error


def check_audio_ready(soundfont: Optional[Path] = None) -> Path:
    """Check local modules and a SoundFont, without network or audio playback."""
    try:
        importlib.import_module("numpy")
        importlib.import_module("music21")
    except (ImportError, OSError) as error:
        raise _setup_error("music21 or NumPy is missing or incompatible") from error
    load_synth_module()
    selected = Path(soundfont).expanduser() if soundfont is not None else DEFAULT_SOUNDFONT
    if not selected.is_file():
        raise _setup_error(f"the SoundFont file is missing ({selected})")
    try:
        with selected.open("rb") as source:
            header = source.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"sfbk":
            raise _setup_error("the SoundFont is not a valid SF2 file")
        if selected.resolve() == DEFAULT_SOUNDFONT.resolve():
            if hashlib.sha256(selected.read_bytes()).hexdigest() != SOUNDFONT_SHA256:
                raise _setup_error("the bundled SoundFont failed its integrity check")
    except OSError as error:
        raise _setup_error(f"the SoundFont could not be read ({selected})") from error
    return selected
