# OmniAnswer

<div align="center">

**A multi-modal, agent-based research assistant that finds, paraphrases, and aggregates high-quality answers across text, images, and video sources.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Project Context

This project was conceived in response to the rapidly evolving landscape of data science and artificial intelligence. As new technologies and methodologies emerge, it becomes increasingly challenging to find clear, concise, and authoritative answers to technical questions. The typical process often involves sifting through a multitude of resources—ranging from Medium articles, Towards Data Science posts, to countless YouTube videos—before finally discovering an article or video that truly clarifies the topic at hand.

**OmniAnswer** aims to streamline this journey by leveraging multi-modal agents and advanced language models to aggregate, paraphrase, and present the most relevant information from across the web, saving users time and effort in their search for knowledge.

---

## 🎯 Key Features

- **Multi-modality Support**: Text, images, and video retrieval and summarization
- **Agent-based Architecture**: Modular agents for each modality with a coordinating orchestration layer using CrewAI
- **Intelligent Paraphrasing**: Automatically rewrites queries to improve search recall for each modality
- **Content Moderation**: Configurable safety guardrails using OpenAI's moderation API to filter unsafe queries
- **Web Interface**: Gradio-based interactive frontend for easy interaction
- **Evaluation Pipeline**: Built-in evaluation with MLflow tracking for similarity metrics (text, image, video)
- **YAML-driven Configuration**: Hydra-compatible configuration system for easy experimentation
- **Docker Support**: Containerized deployment with Docker Compose
- **Cloud Ready**: ECS task definition included for AWS deployment

---

## 🛠️ Tech Stack

<p align="center">
  <img src="assets/python-logo.png" alt="Python" height="40"/>
  <img src="assets/hydra-logo.png" alt="Hydra" height="40"/>
  <img src="assets/crewai-logo.png" alt="CrewAI" height="40"/>
  <img src="assets/google-logo.png" alt="Google APIs" height="40"/>
  <img src="assets/openai-logo.png" alt="OpenAI" height="40"/>
  <img src="assets/gemini-logo.png" alt="Gemini" height="40"/>
  <img src="assets/serpapi-logo.png" alt="SerpApi" height="40"/>
</p>

**Core Technologies:**

- **Python 3.11** - Primary programming language
- **CrewAI** - Multi-agent orchestration framework
- **Hydra & OmegaConf** - Configuration management
- **Gradio** - Web interface framework
- **Pydantic** - Data validation and schemas
- **MLflow** - Experiment tracking and evaluation metrics

**AI/ML Services:**

- **OpenAI** - Content moderation and ChatGPT API calls
- **Google Gemini** - Primary LLM provider (default: gemini-2.5-pro)
- **Tavily** - Text search API
- **Google Custom Search** - Image search
- **SerpApi** - Video/YouTube search

<!-- (Add tech stack logos here: assets/gemini-2.5-pro.jpg, assets/gpt-4o.jpg) -->

---

## 🚀 Quickstart

<!-- (Add quickstart/installation diagram here: assets/dev-env.png) -->

### Prerequisites

- Python 3.11+
- Conda (recommended) or venv
- API keys for the services you plan to use (see [Environment Variables](#environment-variables))

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/lydra4/OmniAnswer.git
cd OmniAnswer
```

2. **Create a Python environment and install dependencies:**

**Using conda (recommended):**

```bash
conda env create -f omnianswer-conda-env.yaml
conda activate omnianswer
```

Note: The conda environment file automatically installs both `requirements.txt` and `dev-requirements.txt` via pip.

**Using venv + pip:**

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

pip install -r requirements.txt -r dev-requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
GOOGLE_API_KEY=your_google_api_key_here
SERP_API_KEY=your_serpapi_key_here
```

4. **Run the application:**

**Option A: Launch the Gradio Web Interface (Recommended for interactive use):**

```bash
python src/launch_gradio.py
```

The web interface will be available at `http://localhost:8080`

**Option B: Run the batch pipeline:**

```bash
python src/pipeline.py
```

The pipeline processes queries from the file specified in `config/pipeline.yaml` (default: `./data/questions/queries.txt`).

---

## 🔧 Environment Variables

<!-- (Add environment variables guide image here: assets/env-vars.png) -->

Required environment variables (store in `.env` or export in your shell):

| Variable         | Description                    | Required For                             |
| ---------------- | ------------------------------ | ---------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key                 | Content moderation and ChatGPT API calls |
| `GEMINI_API_KEY` | Google/Gemini API key          | Primary LLM provider                     |
| `TAVILY_API_KEY` | Tavily search API key          | Text search                              |
| `GOOGLE_CSE_ID`  | Google Custom Search Engine ID | Image search                             |
| `GOOGLE_API_KEY` | Google API key                 | Image search                             |
| `SERP_API_KEY`   | SerpApi key                    | Video/YouTube search                     |

Use `python-dotenv` (already included) to automatically load `.env` files in development.

---

## ⚙️ Configuration

<!-- (Add configuration diagram here: create a config-structure.png showing config hierarchy) -->

All runtime configuration lives in `config/`. The repository uses Hydra-style YAML configuration files.

### Main Configuration

- `config/pipeline.yaml` - Main pipeline configuration
  - Model selection (`model: gemini-2.5-pro`)
  - Temperature settings
  - Evaluation toggle
  - Gradio UI settings

### Agent Configurations

- `config/agent/text_agent.yaml` - Text search agent settings
- `config/agent/image_agent.yaml` - Image search agent settings
- `config/agent/video_agent.yaml` - Video search agent settings
- `config/agent/modality_agent.yaml` - Modality selection agent
- `config/agent/paraphrase_agent.yaml` - Query paraphrasing agent

### Other Configurations

- `config/logging.yaml` - Logging configuration (console and file handlers)
- `config/evaluation/evaluation.yaml` - Evaluation pipeline settings

Example configuration snippet:

```yaml
model: gemini-2.5-pro
temperature: 1.5
moderation_model: omni-moderation-latest
questions: "./data/questions/queries.txt"
evaluate: True
```

---

## 📁 Project Structure

<!-- (Add architecture diagram here: assets/architecture.png) -->

```
OmniAnswer/
├── assets/                        # Diagrams, logos, and images
│   ├── architecture.png           # System architecture diagram
│   ├── architecture.drawio        # Draw.io source for architecture
│   ├── dev-env.png                # Development environment setup
│   ├── dev-workflow.png           # Development workflow diagram
│   ├── env-vars.png               # Environment variables guide
│   └── *.png, *.jpg               # Technology logos
├── config/                        # YAML configuration files
│   ├── agent/                     # Agent-specific configurations
│   │   ├── text_agent.yaml        # Text search agent using TavilySearchTool
│   │   ├── image_agent.yaml       # Image search agent using Google Custom Search
│   │   ├── video_agent.yaml       # Video search agent using SerpApi
│   │   ├── modality_agent.yaml    # Modality selection agent (text/image/video)
│   │   └── paraphrase_agent.yaml  # Query paraphrasing agent for each modality
│   ├── evaluation/                # Evaluation pipeline config
│   │   └── evaluation.yaml        # MLflow tracking and similarity metrics config
│   ├── logging.yaml               # Logging configuration (console and file handlers)
│   └── pipeline.yaml              # Main pipeline config (model, temperature, evaluation)
├── docker/                        # Docker deployment files
│   ├── docker-compose.yml         # Docker Compose configuration
│   └── omnianswer.Dockerfile      # Docker image definition
├── ecs/                           # AWS ECS deployment config
│   └── task-definition.json       # ECS Fargate task definition
├── mlruns/                        # MLflow experiment tracking data
│   └── [experiment_id]/           # Experiment runs and metrics
├── src/                           # Source code
│   ├── agents/                    # Agent implementations
│   │   ├── base_agent/            # Base agent classes
│   │   │   └── base_agent_task.py # Base task implementation
│   │   ├── image_agent.py         # Image search agent
│   │   ├── modality_agent.py      # Modality selection agent
│   │   ├── paraphrase_agent.py    # Query paraphrasing agent
│   │   ├── text_agent.py          # Text search agent
│   │   └── video_agent.py         # Video search agent
│   ├── crew/                      # CrewAI orchestration
│   │   └── orchestrator.py        # Multi-agent crew coordinator
│   ├── evaluation/                # Evaluation pipeline
│   │   └── evaluation_pipeline.py # Similarity metrics computation
│   ├── frontend/                  # Gradio web interface
│   │   └── gradio_app.py          # Interactive web UI
│   ├── moderation/                # Content moderation
│   │   └── content_moderator.py   # OpenAI moderation API wrapper
│   ├── schemas/                   # Pydantic schemas
│   │   └── schemas.py             # Data models and validation
│   ├── tools/                     # Search tools
│   │   ├── image_search.py        # Google Custom Search integration
│   │   └── video_search.py        # SerpApi YouTube search integration
│   ├── utils/                     # Utility functions
│   │   ├── general_utils.py       # LLM loading, MLflow setup
│   │   └── pipeline_utils.py      # Component initialization, file processing
│   ├── launch_gradio.py           # Gradio launcher entrypoint
│   └── pipeline.py                # Batch pipeline entrypoint
├── tests/                         # Unit tests
│   ├── agents/                    # Agent tests
│   │   ├── multi_modality/
│   │   │   └── test_modality_agent.py
│   │   └── test_base_agent.py
│   └── conftest.py                # Pytest configuration
├── requirements.txt               # Runtime dependencies
├── dev-requirements.txt           # Development dependencies (testing, linting)
├── omnianswer-conda-env.yaml      # Conda environment definition
└── README.md
```

<!-- (Add development workflow diagram here: assets/dev-workflow.png) -->

---

## 💻 Usage Examples

### Interactive Web Interface

Launch the Gradio interface:

```bash
python src/launch_gradio.py
```

Then open your browser to `http://localhost:8080` and start asking questions. The interface will:

1. Moderate your query for safety
2. Determine the best learning modalities (text, image, video)
3. Paraphrase queries for each modality
4. Search and return relevant URLs

### Batch Processing

Run the pipeline on a file of queries:

```bash
python src/pipeline.py
```

The pipeline will:

- Process each query from the configured questions file
- Run content moderation
- Select appropriate modalities
- Generate paraphrased queries
- Execute searches across modalities
- Optionally run evaluation metrics (if `evaluate: True` in config)

### Programmatic Usage

```python
from src.utils.pipeline_utils import init_components
from omegaconf import DictConfig
import logging

# Initialize components
cfg = # Load your config
logger = logging.getLogger(__name__)
content_moderator, modality_agent, paraphrase_agent, orchestrator = init_components(
    cfg=cfg, logger=logger
)

# Process a query
query = "How do transformer attention masks work?"
content_moderator.moderate_query(query=query)
modalities = modality_agent.run_query(query=query)
paraphrased_queries = paraphrase_agent.run_query(query=query, modalities=modalities)
result_dict = orchestrator.run(query=query, paraphrase_queries=paraphrased_queries)

print(result_dict)
```

---

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
cd docker
docker-compose up --build
```

The application will be available at `http://localhost:8080`.

### Build Docker Image Manually

```bash
docker build -f docker/omnianswer.Dockerfile -t omnianswer:latest .
docker run -p 8080:8080 --env-file .env omnianswer:latest
```

---

## ☁️ AWS ECS Deployment

The project includes an ECS task definition for deployment on AWS Fargate. See `ecs/task-definition.json` for configuration details.

**Key settings:**

- Network mode: `awsvpc`
- CPU: 1024 (1 vCPU)
- Memory: 2048 MB
- Port: 8080
- Health check: HTTP endpoint on port 8080

---

## 🧪 Evaluation

The evaluation pipeline computes similarity metrics for recommendations:

- **Text Similarity**: BERTScore for text recommendations
- **Image Similarity**: CLIP Score for image recommendations
- **Video Similarity**: X-CLIP Score for video recommendations

Results are tracked in MLflow. To enable evaluation, set `evaluate: True` in `config/pipeline.yaml`.

View MLflow results:

```bash
mlflow ui
```

Then open `http://localhost:5000` to view experiment runs and metrics.

---

## 🧪 Development

### Code Formatting and Linting

Format code with Black:

```bash
black .
```

Lint with Ruff:

```bash
ruff check .
ruff format .
```

Lint with Pylint:

```bash
pylint src
```

### Running Tests

Run the test suite:

```bash
pytest -q
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

---

## 📊 Architecture

<!-- (Add architecture diagram here: assets/architecture.png) -->

OmniAnswer follows a multi-agent architecture:

1. **Content Moderator**: Validates queries for safety using OpenAI's moderation API
2. **Modality Agent**: Determines which modalities (text, image, video) are best suited for a query
3. **Paraphrase Agent**: Rewrites the original query optimized for each selected modality
4. **Orchestrator**: Coordinates specialized agents (text, image, video) using CrewAI
   - **Text Agent**: Searches using Tavily API
   - **Image Agent**: Searches using Google Custom Search
   - **Video Agent**: Searches using SerpApi (YouTube)

The orchestrator aggregates results from all agents and returns structured recommendations.

---

## 🐛 Troubleshooting

### Missing API Key Errors

Ensure all required environment variables are set and visible to the running process. Check that your `.env` file is in the src folder.

### Configuration Not Picked Up

Confirm you're running the pipeline from the project root so relative `config/` paths resolve correctly. Hydra expects to be run from the project root.

### Import Errors

Make sure you've activated your conda/venv environment and installed all dependencies:

```bash
pip install -r requirements.txt -r dev-requirements.txt
```

### Port Already in Use

If port 8080 is already in use, modify the port in `src/frontend/gradio_app.py` or set the `GRADIO_SERVER_PORT` environment variable.

---

## 🤝 Contributing

Contributions are welcome! Here's a suggested workflow:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run tests and linters locally:
   ```bash
   pytest -q
   black .
   ruff check .
   ```
5. Commit your changes: `git commit -m "Add your feature"`
6. Push to your fork: `git push origin feat/your-feature`
7. Open a pull request describing your changes

Please follow existing coding standards and include tests for new behavior.

---

## 📝 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- **CrewAI** - Multi-agent orchestration framework
- **Hydra & OmegaConf** - Configuration management
- **OpenAI** - Content moderation API
- **Google Gemini** - Language model provider
- **Tavily** - Text search API
- **SerpApi** - Video search API
- **Google Custom Search** - Image search
- **MLflow** - Experiment tracking
- **Gradio** - Web interface framework

---

## 📧 Contact

If you have questions or want to collaborate, open an issue or PR on the repository:

**Repository**: https://github.com/lydra4/OmniAnswer

---

## 📚 Additional Resources

- Check `config/` for detailed configuration options
- See `tests/` for example usage patterns
- Review `src/` for implementation details
- MLflow UI for experiment tracking and evaluation metrics
