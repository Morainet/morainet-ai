# Contributing to Morainet AI

Thanks for your interest! This guide covers local setup, the checks CI runs, and
how to add common extensions.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+.

> **macOS / Homebrew Python note**: if `import` fails with a `pyexpat` /
> `libexpat` symbol error, your Homebrew Python links an outdated system
> libexpat. Run commands with:
> `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`
> (already wired into `.vscode/` and a local `.env` for editor runs).

## The checks (same as CI)

```bash
ruff check morainet tests     # lint
mypy morainet                 # strict type check
pytest --cov=morainet         # tests + coverage gate (>= 80%)
```

All three must pass before a PR is merged. CI runs them on Python 3.11 and 3.12.

## Tests

- Unit tests run offline using `MockProvider` — no API keys, no network.
- **Live tests** hit real endpoints and are excluded by default. Run them with
  credentials set:

  ```bash
  MORAINET_OPENAI_API_KEY=sk-...    pytest -m live   # one provider
  pytest -m live                                     # all available; others skip
  ```

  Live tests self-skip when their credential / service is missing.

## Conventions

- Conventional Commits: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`.
- Full type annotations; `mypy --strict` must pass.
- Public APIs need a docstring; new features need tests.
- Keep the core vendor-neutral: a new provider's SDK/dep goes in an optional
  extra (e.g. `[chroma]`, `[mcp]`), never in core dependencies.

## Adding an extension

| Extension | How |
| --- | --- |
| Provider | Subclass `Provider`, implement `chat` (+ `stream`); add a live test |
| Tool | `@tool`-decorate a function, or register via entry point |
| VectorStore | Subclass `VectorStore`, implement `upsert` / `search` |
| Strategy | Subclass `ReasoningStrategy`, implement `run` |
| Hook | Subclass `Hook`, override the callbacks you need |

Third-party packages can also expose extensions via entry points under
`morainet.providers` / `morainet.tools` / `morainet.memory` / `morainet.strategies`
(see `docs/architecture.md` §14).
