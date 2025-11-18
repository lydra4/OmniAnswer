# OmniAnswer

## Project Context

This project was conceived in response to the rapidly evolving landscape of data science and artificial intelligence. As new technologies and methodologies emerge, it becomes increasingly challenging to find clear, concise, and authoritative answers to technical questions. The typical process often involves sifting through a multitude of resources—ranging from Medium articles, Towards Data Science posts, to countless YouTube videos—before finally discovering an article or video that truly clarifies the topic at hand. OmniAnswer aims to streamline this journey by leveraging multi-modal agents and advanced language models to aggregate, paraphrase, and present the most relevant information from across the web, saving users time and effort in their search for knowledge.

## Tech Stack

<p align="center">
  <img src="assets/python-logo.png" alt="Python" height="40"/>
  <img src="assets/hydra-logo.png" alt="Hydra" height="40"/>
  <img src="assets/crewai-logo.png" alt="Agno" height="40"/>
  <img src="assets/google-logo.png" alt="Google APIs" height="40"/>
  <img src="assets/openai-logo.png" alt="OpenAI" height="40"/>
  <img src="assets/gemini-logo.png" alt="Gemini" height="40"/>
  <img src="assets/serpapi-logo.png" alt="SerpApi" height="40"/>

## OmniAnswer

OmniAnswer is a multi-modal, agent-based research assistant designed to find, paraphrase, and aggregate high-quality answers across text, images, and video sources. It combines modular agents (text, image, video, paraphrase) and a team orchestration layer to produce concise, multi-modal responses to user queries while enforcing safety guardrails.

Key goals:
- Reduce research time by surfacing the most relevant resources.
- Optimize queries per modality via paraphrasing.
- Provide a configurable, extensible agent framework for experimentation.

---

## Quick links

- Repository: https://github.com/lydra4/OmniAnswer
- Configs: `config/`
- Source: `src/` (agents, teams, utils)

---

## Features

- Multi-modality: text, images, and video retrieval and summarization.
- Agent-based architecture: modular agents for each modality and a coordinating team layer.
- Paraphrasing: rewrite queries to improve search recall for each modality.
- Safety: configurable guardrails to filter or reject unsafe queries.
- YAML-driven configuration and Hydra-compatible defaults.

---

## Quickstart

1. Clone the repository:

```bash
git clone https://github.com/lydra4/OmniAnswer.git
cd OmniAnswer
```

2. Create a Python environment and install dependencies. (Conda recommended, but venv/pip works.)

Using conda (recommended):

```bash
conda env create -f omnianswer-conda-env.yaml
conda activate omnianswer
pip install -r requirements.txt -r dev-requirements.txt
```

Using venv + pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r dev-requirements.txt
```

3. Create a `.env` file or export required environment variables (examples below).

4. Run the pipeline (note: this repository currently provides `src/pipeline.py` as the main runnable pipeline):

```bash
python -m src.pipeline
```

If you previously saw instructions referencing `src/main.py`, this repository uses `src/pipeline.py` as the entrypoint — update as needed.

---

## Environment variables

Recommended variables (store in `.env` or export in your shell):

- OPENAI_API_KEY      — OpenAI API key (if using OpenAI models)
- GEMINI_API_KEY      — Google/Gemini API key (if using Gemini)
- GOOGLE_CSE_ID       — Google Custom Search ID (for search)
- SERP_API_KEY        — SerpApi key for YouTube/video search
- AGNO_API_KEY        — Agno framework key (if required)

Use `python-dotenv` or your preferred approach to load `.env` in development.

---

## Configuration

All runtime configuration lives in `config/`. The repo uses Hydra-style defaults in YAML.

- Inspect `config/` to set model selection, temperature, agent parameters, and logging.
- `config/logging.yaml` configures console and file handlers (logs/).

Example snippet (see `config/` for full files):

```yaml
model: gemini-2.5-pro
temperature: 1.0
```

---

## Project layout (important files)

`assets/` — diagrams and logos used by the README and documentation.

`config/` — YAML configuration for agents, teams, and logging.

`src/` — implementation. Notable modules:
- `src/pipeline.py` — pipeline/entrypoint for running queries and agent teams.
- `src/agents/` — agent implementations (paraphrase, text, image, video, etc.).
- `src/teams/` or `src/crew/` — orchestration logic for multi-modal teams.
- `src/utils/` — utility helpers.

`requirements.txt` — runtime dependencies.
`dev-requirements.txt` — development tools (linters, formatters, test tools).

---

## Usage examples

Interactive example (pseudo):

```python
from src.pipeline import run_query

result = run_query("How do transformer attention masks work?")
print(result.summary)
# result will include text, and lists of image/video candidates when applicable
```

CLI example:

```bash
python -m src.pipeline --query "best practices for retraining LLMs"
```

Note: exact function/CLI flags depend on the pipeline implementation; inspect `src/pipeline.py` for current parameters.

---

## Development

- Formatting: `black` (run `black .`)
- Linting: `ruff` / `pylint` (run `ruff .` or `pylint src`)
- Pre-commit: configured via `pre-commit` (if present)

Run unit tests under `tests/` with pytest:

```bash
pytest -q
```

If tests fail after edits, run linters and fix warnings before submitting PRs.

---

## Contributing

Contributions are welcome. A suggested workflow:

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Run tests and linters locally.
4. Open a pull request describing your changes and any migration notes.

Please follow existing coding standards and include tests for new behavior.

---

## Troubleshooting

- Missing API key errors: ensure environment variables are set and visible to the running process.
- Config not picked up: confirm you're running the pipeline from the project root so relative `config/` paths resolve.
- If an entrypoint like `src/main.py` is referenced in docs but not present, use `src/pipeline.py` or check the source to confirm the current entrypoint.

---

## Tests & Quality gates

Run the test suite:

```bash
pytest -q
```

Run linters and fixers:

```bash
ruff format .
black .
ruff check .
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- Agno agent framework
- Hydra & OmegaConf for configuration
- Guardrails for safety constraints
- SerpApi, Google Search/Images for search integrations

---

## Contact

If you have questions or want to collaborate, open an issue or PR on the repository: https://github.com/lydra4/OmniAnswer

---

Notes/assumptions:
- The repository's runnable entrypoint appears to be `src/pipeline.py` (no `src/main.py` found). I updated examples and Quickstart accordingly — if you'd prefer a different entrypoint, tell me and I will swap references.
