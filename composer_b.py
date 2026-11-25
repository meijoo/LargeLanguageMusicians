"""Import and launch Composer B from its named folder."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().with_name('Composer B'))]

if __name__ == "__main__":
    from composer_b.agent import main
    raise SystemExit(main())
