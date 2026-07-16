# Contributing

Thank you for your interest in improving this repository.

## Development setup

```bash
git clone https://github.com/REPLACE_ORG/sMRI_pipeline.git
cd sMRI_pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
# Optional MSN / GCN extras:
pip install -e ".[msn,dev]"
```

## Guidelines

- Do **not** commit patient-level or identifiable clinical/MRI data.
- Keep notebooks under `notebooks/` with English filenames and cleared heavy outputs.
- Prefer relative paths under `./data/` as described in `data/README.md`.
- Add or update tests under `tests/` for non-trivial code changes.
- Open a pull request with a short description of the motivation and how you tested the change.

## Code of conduct

Be respectful and constructive in discussions and reviews.
