# Contributing to GeoPrep

Thanks for your interest in contributing! GeoPrep is a preprocessing library only — it does not train or run models, so please keep contributions scoped to data discovery, preprocessing, alignment, feature generation, temporal/graph construction, validation, and dataset export.

## Development Setup

```bash
git clone https://github.com/<your-username>/geoprep.git
cd geoprep
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

Please add or update tests under `tests/` for any behavioral change. Tests that exercise numerical correctness (e.g. comparing a vectorized implementation's output against a known-good reference) are especially valuable in this codebase, since a fast-but-wrong implementation is worse than a slow-but-correct one.

## Guidelines

- **No per-pixel Python loops.** Every raster operation should be vectorized via NumPy, OpenCV, or SciPy. If you find yourself writing `for row in array: for value in row:`, there's almost always a vectorized equivalent.
- **Watch for NumPy truthiness bugs.** `if some_array:` raises `ValueError` for arrays with more than one element. Use `is None` / `is not None`, or an explicit reduction (`.any()`, `.all()`), never bare truthiness on an array that might have more than one element.
- **Keep public APIs stable.** If you need to change a function signature, prefer adding an optional keyword argument with a backward-compatible default over renaming or removing existing parameters.
- **Match the dataclass contracts.** If a model field is typed `np.ndarray`, don't hand it a Python list — several past bugs in this codebase came from that exact mismatch.
- **Follow PEP8** and use type hints throughout.
- **Write docstrings** for public classes and functions.
- **Raise the existing custom exceptions** (see `geoprep.core.exceptions`) rather than bare `ValueError`/`RuntimeError` for domain-level failures, so callers can catch `GeoPrepError` uniformly.
- **Keep modules single-responsibility.** Preprocessing shouldn't know about graphs; graph construction shouldn't know about temporal sequences, etc.

## Submitting Changes

1. Fork the repo and create a feature branch.
2. Make your change, with tests.
3. Run `pytest` locally and confirm it passes.
4. Open a Pull Request describing what changed and why. If you're fixing a numerical bug, include a before/after comparison if possible (this project has a history of subtle vectorization bugs that only show up numerically, not as exceptions).

## Code of Conduct

Please read `CODE_OF_CONDUCT.md` before participating.
