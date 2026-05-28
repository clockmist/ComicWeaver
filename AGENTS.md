# Repository Guidelines

## Project Structure & Module Organization

ComicWeaver uses a Python `src/` layout. Application code lives in `src/comicweaver/`: `core/` contains shared schemas, state, base classes, and exceptions; `agents/` contains the script, character, storyboard, image, layout, and reviewer agents; `orchestrator/` coordinates workflow execution; `review/` stores review decisions and rubric data; `storage/` handles persistence; `ui/` contains the Gradio interface; and `interaction/` holds adapters. Tests are under `tests/unit/` and `tests/integration/`. Design notes and agent interface docs are in `docs/`, including `docs/agents/README.md`.

## Build, Test, and Development Commands

- `python -m venv .venv` then `source .venv/bin/activate`: create and activate a local environment. On Windows use `.venv/Scripts/activate`.
- `pip install -e .`: install ComicWeaver in editable mode with runtime dependencies.
- `pip install -e ".[dev]"`: install development tools such as `pytest`, `ruff`, and `mypy`.
- `python -m comicweaver` or `comicweaver`: launch the Gradio UI at `http://localhost:7860`.
- `pytest`: run all tests configured by `pyproject.toml`.
- `ruff check src tests`: run lint checks.
- `mypy src`: run static type checks.

## Coding Style & Naming Conventions

Target Python 3.10 or newer. Use 4-space indentation, type hints for public interfaces, and concise docstrings for modules or non-obvious behavior. Follow the existing package naming style: lowercase modules, `snake_case` functions and variables, `PascalCase` classes, and agent files named like `script_agent.py`. Ruff is configured for a 100-character line length, import sorting, pyupgrade, bugbear, and pep8-naming checks; avoid adding broad lint suppressions.

## Testing Guidelines

Use `pytest`. Place unit tests in `tests/unit/test_*.py` and workflow tests in `tests/integration/test_*.py`. Name tests by observable behavior, for example `test_reviewer_blocks_low_quality_panel`. Add or update tests when changing schemas, agent outputs, orchestration flow, storage behavior, or UI-facing contracts. Run `pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent history uses short subject lines such as `feat:add docs` and `feat: initial framework with Mock agents and Gradio UI`. Prefer concise, imperative messages with an optional conventional prefix, for example `fix: validate storyboard panel count`. Pull requests should include a brief summary, tests run, linked issue or course task when applicable, and screenshots or short recordings for visible Gradio UI changes. Note any mock behavior, missing model backends, or configuration assumptions.

## Security & Configuration Tips

Do not commit local virtual environments, generated outputs, API keys, model checkpoints, or large datasets. Keep real LLM and diffusion backends behind optional dependencies and configuration so the default mock workflow remains lightweight and runnable without GPU access.
