# OmniAnswer

<div align="center">

**A multi-modal, agent-based research assistant that finds, paraphrases, and aggregates high-quality answers across text, images, and video sources.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📑 Table of Contents

- [Project Context](#-project-context)
- [Key Features](#-key-features)
- [Quick Demo](#-quick-demo)
- [Tech Stack](#️-tech-stack)
- [Quickstart](#-quickstart)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Environment Variables](#-environment-variables)
- [Configuration](#️-configuration)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Docker Deployment](#-docker-deployment)
- [AWS ECS Deployment](#️-aws-ecs-deployment)
- [Evaluation](#-evaluation)
- [Development](#-development)
- [Architecture](#-architecture)
  - [Evaluation Architecture](#evaluation-architecture)
- [Troubleshooting](#-troubleshooting)
  - [Common Issues](#common-issues)
- [FAQ](#-frequently-asked-questions)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)
- [Additional Resources](#-additional-resources)

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

## 🎬 Quick Demo

Here's what happens when you ask OmniAnswer a question:

![Demo](assets/gradio_app.gif)

**Example Query:** _"How does gradient descent work?"_

### Step-by-Step Process:

1. **🔒 Content Moderation**

   - Query is checked for safety using OpenAI's moderation API
   - ✅ Query passes moderation

2. **🎯 Modality Selection**

   - OmniAnswer determines the best learning modalities for your query
   - Selected modalities: **Text** + **Video**

3. **🔄 Query Paraphrasing**

   - Original query is optimized for each modality:
     - **Text**: "gradient descent algorithm explanation mathematical optimization"
     - **Video**: "gradient descent visualization tutorial step by step"

4. **🔍 Multi-Modal Search**

   - **Text Agent** searches using Tavily API → Returns 5 relevant articles
   - **Video Agent** searches using SerpApi (YouTube) → Returns 3 tutorial videos

5. **📊 Results Aggregation**
   - All results are combined and presented with:
     - Source URLs
     - Relevance scores (if evaluation is enabled)
     - Modality-specific metadata

### Try It Yourself:

```bash
python src/launch_gradio.py
```

Then visit `http://localhost:8080` and ask any question!

**More Example Queries:**

- _"Explain neural network backpropagation"_ → Text + Video
- _"Show me data visualization best practices"_ → Image + Text
- _"How to implement gradient descent in Python?"_ → Video + Text

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

<p align="center">
  <img src="assets/llm-gemini-2.5-pro.jpg" alt="gemini-2.5-pro" height="200"/>
  <img src="assets/llm-gpt-4o.jpg" alt="gpt-4o" height="200"/>
</p>

---

## 🚀 Quickstart

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

Create a `.env` file in the src folder with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
GOOGLE_API_KEY=your_google_api_key_here
SERP_API_KEY=your_serpapi_key_here
```

4. **Launch the Gradio Web Interface:**

```bash
python src/launch_gradio.py
```

The web interface will be available at `http://localhost:8080`. You can interact with the agents through the UI to ask questions and receive multi-modal search results.

---

## 🔧 Environment Variables

**What it shows**: The environment variables guide provides a visual overview of all required API keys and configuration variables needed to run OmniAnswer. It explains which services each variable connects to (OpenAI, Gemini, Tavily, Google Custom Search, SerpApi) and how they're used throughout the system. This diagram helps developers quickly identify which API keys they need to obtain and configure before running the application.

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

<p align="center">
  <img src="assets/config-structure.png" alt="Configuration Structure" height="400"/>
</p>

**What it shows**: The configuration structure diagram (`assets/config-structure.png`) visualizes the hierarchical organization of YAML configuration files in the `config/` directory. It shows how the main `pipeline.yaml` file references agent-specific configurations, evaluation settings, and logging configurations. This diagram helps developers understand how Hydra merges configurations and where to modify settings for different components of the system.

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

<!-- Architecture diagram is shown in the Architecture section below -->

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

<p align="center">
  <img src="assets/dev-workflow.png" alt="Development Workflow" height="400"/>
</p>

**What it shows**: The development workflow diagram (`assets/dev-workflow.png`) illustrates the complete software development lifecycle for contributing to OmniAnswer. It shows the iterative process of editing code, running tests, applying linting and formatting tools (Black, Ruff, Pylint), committing changes, and deploying updates. This diagram helps contributors understand the recommended workflow for maintaining code quality and ensuring changes are properly tested before integration.

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

---

## 🐳 Docker Deployment

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

<p align="center">
  <img src="assets/architecture.jpg" alt="System Architecture" height="500"/>
</p>

**What it shows**: The architecture diagram (`assets/architecture.jpg`) provides a high-level overview of the OmniAnswer system's multi-agent architecture. It visualizes the flow of data from user queries through content moderation, modality selection, query paraphrasing, and multi-modal search orchestration. The diagram shows how different agents interact and how results are aggregated and returned to users. This helps developers understand the system's component relationships and data flow.

OmniAnswer follows a multi-agent architecture:

1. **Content Moderator**: Validates queries for safety using OpenAI's moderation API
2. **Modality Agent**: Determines which modalities (text, image, video) are best suited for a query
3. **Paraphrase Agent**: Rewrites the original query optimized for each selected modality
4. **Orchestrator**: Coordinates specialized agents (text, image, video) using CrewAI
   - **Text Agent**: Searches using Tavily API
   - **Image Agent**: Searches using Google Custom Search
   - **Video Agent**: Searches using SerpApi (YouTube)

The orchestrator aggregates results from all agents and returns structured recommendations.

### Evaluation Architecture

The evaluation pipeline is integrated into the main architecture and provides quantitative assessment of recommendation quality:

1. **Evaluation Pipeline** (`src/evaluation/evaluation_pipeline.py`): Computes modality-specific similarity metrics after the orchestrator returns results

   - **Text Evaluation**: Uses BERTScore to measure semantic similarity between paraphrased queries and scraped article content
     - Scrapes text from URLs using ScraperAPI
     - Extracts article content (preferring `<article>` tags, falling back to `<body>`)
     - Compares query paraphrases against scraped text using BERT-base-uncased embeddings
   - **Image Evaluation**: Uses CLIP Score to measure visual-text alignment
     - Downloads images from URLs
     - Computes CLIP-based similarity between image embeddings and query paraphrase embeddings
     - Uses OpenAI's CLIP-ViT-Base-Patch32 model
   - **Video Evaluation**: Uses X-CLIP Score to measure video-text alignment
     - Downloads video segments (default: 5 seconds) using yt-dlp and ffmpeg
     - Samples evenly-spaced frames based on model requirements (X-CLIP-base-patch32)
     - Computes cosine similarity between video and text embeddings in the X-CLIP embedding space

2. **MLflow Integration**: All evaluation metrics are automatically logged to MLflow for experiment tracking

   - Tracks text similarity, image similarity, and video similarity scores per query
   - Records model configuration (LLM name, temperature) and selected modalities
   - Enables comparison across different model configurations and hyperparameters
   - Stores results in `mlruns/` directory for analysis via MLflow UI

3. **Evaluation Configuration**: Controlled via `config/evaluation/evaluation.yaml`
   - Configurable model selection for each modality (BERT, CLIP, X-CLIP variants)
   - Adjustable parameters (text word count, video duration)
   - MLflow experiment naming and directory configuration

The evaluation pipeline runs automatically when `evaluate: True` is set in `config/pipeline.yaml`, allowing for systematic assessment of recommendation quality across different model configurations and query types.

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

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'crewai'"

**Symptoms:**

```
ModuleNotFoundError: No module named 'crewai'
```

**Solution:**

1. Ensure you've activated your conda/venv environment:

   ```bash
   # For conda:
   conda activate omnianswer

   # For venv:
   # Windows:
   .venv\Scripts\activate
   # Unix/MacOS:
   source .venv/bin/activate
   ```

2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt -r dev-requirements.txt
   ```

#### Issue: "API rate limit exceeded" or "429 Too Many Requests"

**Symptoms:**

- API calls fail with rate limit errors
- Search results are empty or incomplete

**Solution:**

1. Check your API key quotas and usage limits for each service:

   - OpenAI: Check [OpenAI Usage Dashboard](https://platform.openai.com/usage)
   - Gemini: Check [Google Cloud Console](https://console.cloud.google.com/)
   - Tavily: Check [Tavily Dashboard](https://tavily.com/dashboard)
   - SerpApi: Check [SerpApi Dashboard](https://serpapi.com/dashboard)

2. Implement rate limiting in your configuration or wait before retrying

3. Consider upgrading your API plan if you need higher rate limits

#### Issue: "Evaluation metrics are 0.0" or evaluation fails

**Symptoms:**

- MLflow shows similarity scores of 0.0
- Evaluation pipeline crashes with model loading errors

**Solution:**

1. Ensure evaluation models are downloaded:

   - Text evaluation requires BERT model (auto-downloaded)
   - Image evaluation requires CLIP model (auto-downloaded)
   - Video evaluation requires X-CLIP model and ffmpeg

2. For video evaluation, install required dependencies:

   ```bash
   # Install ffmpeg (varies by OS)
   # Windows: Download from https://ffmpeg.org/download.html
   # macOS: brew install ffmpeg
   # Linux: sudo apt-get install ffmpeg

   pip install yt-dlp
   ```

3. Check that you have sufficient disk space and memory for model downloads

#### Issue: "HydraConfigNotFound" or configuration errors

**Symptoms:**

- Configuration files not found
- Hydra initialization errors

**Solution:**

1. Ensure you're running from the project root directory:

   ```bash
   cd OmniAnswer
   python src/launch_gradio.py
   ```

2. Verify `config/` directory exists and contains all required YAML files

3. Check that configuration file syntax is valid YAML

#### Issue: "Connection timeout" or network errors

**Symptoms:**

- API calls timeout
- Search results fail to load

**Solution:**

1. Check your internet connection
2. Verify API keys are valid and not expired
3. Check if any firewall or proxy is blocking API requests
4. For Google Custom Search, ensure your CSE ID is correctly configured

#### Issue: "Gradio interface not loading" or blank page

**Symptoms:**

- Browser shows blank page at `http://localhost:8080`
- Connection refused errors

**Solution:**

1. Check if the Gradio server started successfully (look for startup messages)
2. Verify port 8080 is not in use by another application:

   ```bash
   # Windows:
   netstat -ano | findstr :8080
   # Unix/MacOS:
   lsof -i :8080
   ```

3. Try a different port by setting environment variable:

   ```bash
   set GRADIO_SERVER_PORT=8081  # Windows
   export GRADIO_SERVER_PORT=8081  # Unix/MacOS
   ```

4. Check browser console for JavaScript errors

#### Issue: "Content moderation blocking valid queries"

**Symptoms:**

- Legitimate queries are flagged as unsafe
- Moderation API returns false positives

**Solution:**

1. Review your query - some technical terms may trigger moderation
2. Check moderation settings in `config/pipeline.yaml`
3. Consider adjusting the moderation model or threshold if needed
4. For development, you can temporarily disable moderation (not recommended for production)

---

## ❓ Frequently Asked Questions

### General Questions

**Q: Do I need all API keys to use OmniAnswer?**  
A: No, you only need the API keys for the modalities you want to use. However, you must have at least one search API key (Tavily, Google Custom Search, or SerpApi) and the LLM provider key (Gemini). OpenAI API key is required for content moderation.

**Q: Can I use OmniAnswer without the web interface?**  
A: Yes! You can use the batch pipeline by running `python src/pipeline.py` with a queries file. Check `config/pipeline.yaml` for configuration options.

**Q: What programming languages does OmniAnswer support?**  
A: Currently, OmniAnswer is built with Python 3.11+ and supports queries in English. The system can search for content in any language, but the query processing and agent interactions are optimized for English.

### Configuration Questions

**Q: Can I use a different LLM provider instead of Gemini?**  
A: Yes, you can modify `config/pipeline.yaml` to change the model. OmniAnswer uses CrewAI, which supports multiple LLM providers including OpenAI, Anthropic, and others. Check the [CrewAI documentation](https://docs.crewai.com/) for supported providers.

**Q: How do I disable evaluation?**  
A: Set `evaluate: False` in `config/pipeline.yaml`. This will skip the evaluation pipeline and improve response time, but you won't get similarity metrics.

**Q: Can I customize the number of results per modality?**  
A: Yes, you can modify the agent configurations in `config/agent/` directory. Each agent (text, image, video) has settings for the number of results to return.

### Technical Questions

**Q: Does OmniAnswer require a GPU?**  
A: No, OmniAnswer works on CPU. However, evaluation metrics (BERTScore, CLIP, X-CLIP) will run faster with a GPU. For production use without evaluation, CPU is sufficient.

**Q: How much memory does OmniAnswer need?**  
A: Minimum 4GB RAM is recommended. If you're running evaluation, 8GB+ is recommended. The LLM models are loaded into memory, and evaluation models require additional memory.

**Q: Can I run OmniAnswer in a Docker container?**  
A: Yes! See the [Docker Deployment](#-docker-deployment) section. The project includes a Dockerfile and docker-compose.yml for easy containerized deployment.

**Q: How do I deploy OmniAnswer to AWS?**  
A: The project includes an ECS task definition in `ecs/task-definition.json`. See the [AWS ECS Deployment](#️-aws-ecs-deployment) section for details.

### Evaluation Questions

**Q: What do the similarity scores mean?**  
A: Similarity scores range from 0.0 to 1.0, where higher scores indicate better relevance:

- **Text Similarity (BERTScore)**: Measures semantic similarity between query and article content
- **Image Similarity (CLIP Score)**: Measures visual-text alignment
- **Video Similarity (X-CLIP Score)**: Measures video-text alignment

**Q: How do I view evaluation results?**  
A: Run `mlflow ui` and open `http://localhost:5000` in your browser. You'll see all experiment runs, metrics, and can compare different configurations.

**Q: Can I evaluate my own queries?**  
A: Yes! Create a text file with one query per line and set `questions: "./path/to/your/queries.txt"` in `config/pipeline.yaml`. Then run the batch pipeline.

### API and Service Questions

**Q: What are the API rate limits?**  
A: Rate limits depend on your subscription with each service:

- **OpenAI**: Varies by plan (check your dashboard)
- **Gemini**: Free tier has limits, paid plans have higher quotas
- **Tavily**: Check [Tavily pricing](https://tavily.com/pricing)
- **Google Custom Search**: 100 queries/day free, more with paid plans
- **SerpApi**: Varies by plan (check [SerpApi pricing](https://serpapi.com/pricing))

**Q: How much do the API services cost?**  
A: Costs vary by provider and usage. Most services offer free tiers for development. Check each provider's pricing page for current rates.

**Q: Can I use my own search APIs?**  
A: Yes, you can extend the search tools in `src/tools/` to integrate with your preferred search APIs. The agent architecture is modular and supports custom tools.

### Development Questions

**Q: How do I contribute to OmniAnswer?**  
A: See the [Contributing](#-contributing) section. Fork the repository, create a feature branch, make your changes, run tests, and submit a pull request.

**Q: What code style should I follow?**  
A: The project uses Black for formatting and Ruff/Pylint for linting. Run `black .` and `ruff check .` before committing. See the [Development](#-development) section for details.

**Q: How do I run the tests?**  
A: Run `pytest -q` from the project root. For coverage reports, use `pytest --cov=src --cov-report=html`.

**Q: Can I add support for additional modalities (e.g., audio)?**  
A: Yes! The architecture is designed to be extensible. You can create a new agent in `src/agents/`, add a search tool in `src/tools/`, and update the modality agent to include your new modality.

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

### Documentation Links

- **[CrewAI Documentation](https://docs.crewai.com/)** - Learn about multi-agent orchestration, agents, crews, and flows
- **[Hydra Documentation](https://hydra.cc/docs/intro/)** - Understand hierarchical configuration composition and command-line overrides
- **[MLflow Python API](https://mlflow.org/docs/latest/api_reference/python_api/index.html)** - Reference for experiment tracking, metrics logging, and model management

### Project Resources

- Check `config/` for detailed configuration options
- See `tests/` for example usage patterns
- Review `src/` for implementation details
- MLflow UI for experiment tracking and evaluation metrics (run `mlflow ui` to access)
