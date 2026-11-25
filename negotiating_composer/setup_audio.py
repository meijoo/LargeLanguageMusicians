"""One-time local setup for automatic ABC/MIDI/WAV export.

TinySoundFont's PyAudio dependency is only needed for its live-playback method.
Install the wheel in a separate local target because this project renders WAV
offline and does not need PyAudio, PortAudio, MuseScore, or a system synthesizer.
"""

import hashlib
from importlib import metadata
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

if __package__:
    from .audio_runtime import (
        DEFAULT_SOUNDFONT, RUNTIME_PACKAGES, RUNTIME_ROOT,
        SOUNDFONT_SHA256, check_audio_ready,
    )
else:
    from audio_runtime import (
        DEFAULT_SOUNDFONT, RUNTIME_PACKAGES, RUNTIME_ROOT,
        SOUNDFONT_SHA256, check_audio_ready,
    )


MUSIC21_VERSION = "9.3.0"
SYNTH_VERSION = "0.3.7"
SOUNDFONT_COMMIT = "d6ad4ed72dce1fd3d67f17b74e08cd7ae7941a96"
SOUNDFONT_SOURCE = f"https://raw.githubusercontent.com/arbruijn/TimGM6mb/{SOUNDFONT_COMMIT}/"
LICENSE_SHA256 = "caf4761ca5e96fd034502a4dfac59f65fce61f3e5b4801b0b149c900fee29f06"


def _base_dependencies_ready() -> bool:
    try:
        numpy_version = metadata.version("numpy").split(".")
        if metadata.version("music21") != MUSIC21_VERSION:
            return False
        if not (numpy_version[0] == "1" and int(numpy_version[1]) >= 26):
            return False
        importlib.import_module("numpy")
        importlib.import_module("music21")
        return True
    except (metadata.PackageNotFoundError, ImportError, OSError, ValueError):
        return False


def _synth_ready() -> bool:
    try:
        installed = next(
            distribution for distribution in metadata.distributions(path=[str(RUNTIME_PACKAGES)])
            if distribution.metadata["Name"] == "tinysoundfont"
        )
        if installed.version != SYNTH_VERSION:
            return False
        if str(RUNTIME_PACKAGES) not in sys.path:
            sys.path.insert(0, str(RUNTIME_PACKAGES))
        importlib.invalidate_caches()
        module = importlib.import_module("tinysoundfont")
        return hasattr(module, "Synth") and hasattr(module, "Sequencer")
    except (StopIteration, ImportError, OSError):
        return False


def _download_verified(filename: str, expected_hash: str) -> None:
    destination = RUNTIME_ROOT / filename
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash:
        return
    print(f"Downloading {filename}...", flush=True)
    request = Request(SOUNDFONT_SOURCE + filename, headers={"User-Agent": "ComposerX-audio-setup"})
    temporary = None
    try:
        with urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(dir=RUNTIME_ROOT, delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Downloaded {filename} failed its SHA-256 integrity check.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def setup_audio() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if not _base_dependencies_ready():
        print("Installing the ABC converter...", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir",
            f"music21=={MUSIC21_VERSION}", "numpy>=1.26,<2",
        ], check=True)
        importlib.invalidate_caches()
    if not _synth_ready():
        print("Installing the offline synthesizer...", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--no-deps",
            "--target", str(RUNTIME_PACKAGES), f"tinysoundfont=={SYNTH_VERSION}",
        ], check=True)
        importlib.invalidate_caches()
        for name in list(sys.modules):
            if name == "tinysoundfont" or name.startswith("tinysoundfont."):
                del sys.modules[name]
    _download_verified(DEFAULT_SOUNDFONT.name, SOUNDFONT_SHA256)
    _download_verified("COPYING.txt", LICENSE_SHA256)
    sources = {
        "soundfont": {
            "name": "TimGM6mb", "creators": ["Tim Brechbill", "David Bolton"],
            "license": "GPL-2.0-only", "repository": "https://github.com/arbruijn/TimGM6mb",
            "commit": SOUNDFONT_COMMIT, "url": SOUNDFONT_SOURCE + DEFAULT_SOUNDFONT.name,
            "sha256": SOUNDFONT_SHA256, "license_sha256": LICENSE_SHA256,
        },
        "synthesizer": {
            "name": "tinysoundfont", "version": SYNTH_VERSION, "license": "MIT",
            "source": "https://github.com/amberwhitehead/tinysoundfont-pybind",
            "mode": "Offline generation; PyAudio is not used.",
        },
        "converter": {"name": "music21", "version": MUSIC21_VERSION},
    }
    (RUNTIME_ROOT / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return check_audio_ready()


def main() -> int:
    try:
        font = setup_audio()
    except Exception as error:
        print(f"Audio setup failed: {error}", file=sys.stderr)
        return 1
    print(f"Audio setup ready. SoundFont: {font}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
