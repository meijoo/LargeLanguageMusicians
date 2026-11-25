"""Import and launch Composer A from its named folder."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().with_name('Composer A'))]

if __name__ == "__main__":
    from composer_a.composer import main
    raise SystemExit(main())
