"""Allocate unique result folders without overwriting previous compositions."""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent
SINGLE_RESULTS = PROJECT_ROOT / 'Single Composition Results'
NEGOTIATED_RESULTS = PROJECT_ROOT / 'Negotiated Composition Results'


def reserve_result(prefix, parent=None):
    if prefix not in ('A', 'B', 'AB'):
        raise ValueError('Composer prefix must be A, B, or AB')
    root = Path(parent) if parent is not None else (NEGOTIATED_RESULTS if prefix == 'AB' else SINGLE_RESULTS)
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r'_(\d+)$')
    while True:
        numbers = [int(match.group(1)) for path in root.iterdir()
                   if (match := pattern.fullmatch(path.name))]
        destination = root / f'{prefix}_{max(numbers, default=0) + 1}'
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
