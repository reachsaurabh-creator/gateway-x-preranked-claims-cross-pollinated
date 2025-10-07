# Gateway X: Preranked Claims Cross-Pollinated

Gateway X: Preranked Claims Cross-Pollinated is a variant of the consensus engine that implements a different algorithm for consensus generation. This approach uses preranked claims that are cross-pollinated between AI engines to reach consensus, while maintaining the same infrastructure and transparency features as the original Gateway X Consensus Engine.

## Features

- **Preranked Claims Algorithm**: Uses a different consensus approach with preranked claims
- **Cross-Pollination Engine**: Implements cross-pollination of claims between AI engines
- **Multi-Engine Orchestration**: Coordinates multiple AI engines to generate diverse candidate answers
- **Dueling Comparisons**: Uses pairwise comparisons with LLM referees to evaluate answer quality
- **BTL Ranking**: Implements Bradley-Terry-Luce model with Minorization-Maximization updates
- **Bootstrap Confidence Intervals**: Provides statistical confidence measures for rankings
- **Convergence Monitoring**: Tracks consensus convergence with multiple stop conditions
- **Transparency Ledger**: Comprehensive logging of all consensus events
- **HTML Reports**: Generates detailed timeline reports with inline SVG visualizations
- **FastAPI Interface**: RESTful API for easy integration

## Architecture

```
gateway-x/
├─ app/
│  ├─ config.py                # Environment-driven configuration
│  ├─ schemas.py               # Pydantic models (QueryIn/Out, TimelineItem, ClaimScore)
│  ├─ prompts.py               # PromptVault with strict JSON validation
│  ├─ duel_scheduler.py        # AsyncAnthropic integration with caching
│  ├─ btl_ranker.py            # MM algorithm + bootstrap confidence intervals
│  ├─ playbook_selector.py     # UCB-based strategy selector
│  ├─ monitors.py              # StopConditionEvaluator (CI-separation + confidence)
│  ├─ ledger.py                # TransparencyLedger for structured logging
│  ├─ orchestrator.py          # Main orchestrator (timeline management)
│  ├─ report.py                # HTML report renderer with inline SVG
│  └─ server.py                # FastAPI application
├─ tests/                      # Comprehensive test suite
└─ gatewayx_single.py          # Single-file prototype for rapid iteration
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd gateway-x

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### 2. Configuration

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` to set your configuration:

```bash
# Use real LLM (requires Anthropic API key)
GX_USE_REAL_LLM=true
ANTHROPIC_API_KEY=your_api_key_here

# Or use mock referee for testing
GX_USE_REAL_LLM=false
```

### 3. Run the Server

#### Option A: Modular Version
```bash
cd gateway-x
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

#### Option B: Single-file Prototype
```bash
uvicorn gatewayx_single:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Submit a consensus query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "budget": 10,
    "confidence_threshold": 0.95
  }'

# Get timeline for a run
curl "http://localhost:8000/timeline/{run_id}"

# Get HTML report
curl "http://localhost:8000/timeline/{run_id}/report"
```

## API Endpoints

### POST `/query`

Submit a query for consensus processing.

**Request Body:**
```json
{
  "query": "Your question here",
  "budget": 12,
  "confidence_threshold": 0.95
}
```

**Response:**
```json
{
  "run_id": "abc123def456",
  "query": "Your question here",
  "best_claim": "The consensus answer",
  "confidence": 0.87,
  "rounds": 8,
  "total_duels": 24,
  "stop_reason": "confidence_threshold"
}
```

### GET `/timeline/{run_id}`

Get the complete timeline for a consensus run as JSON.

### GET `/timeline/{run_id}/report`

Get a formatted HTML report with visualizations.

### GET `/health`

Health check endpoint.

## Configuration

Gateway X is configured via environment variables with the `GX_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `GX_USE_REAL_LLM` | `false` | Use real LLM API vs mock referee |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `GX_DEFAULT_BUDGET` | `12` | Default number of rounds |
| `GX_MAX_BUDGET` | `200` | Maximum allowed rounds |
| `GX_CONFIDENCE_THR` | `0.95` | Default confidence threshold |
| `GX_CI_MIN_ROUNDS` | `6` | Minimum rounds before CI calculation |
| `GX_CI_BOOTSTRAP` | `200` | Bootstrap samples for CI |
| `GX_DUELS_PER_ROUND` | `3` | Duels per consensus round |

See `env.example` for the complete list.

## Algorithm Overview

**Preranked Claims Cross-Pollination Algorithm:**

1. **Initial Round**: Generate diverse candidate answers from multiple engines
2. **Preranking Phase**: Rank initial claims using preliminary scoring mechanisms
3. **Cross-Pollination Rounds**:
   - Select top-ranked claims from each engine
   - Cross-pollinate claims between engines for refinement
   - Generate improved claims based on cross-pollinated insights
   - Update rankings based on cross-pollination results
4. **Iterative Consensus**:
   - Select most informative pairs using entropy-based scoring
   - Run parallel duels with LLM referee
   - Update BTL scores using MM algorithm
   - Compute bootstrap confidence intervals
   - Evaluate convergence conditions
5. **Stop Conditions**:
   - CI separation achieved (statistical significance)
   - Confidence threshold reached
   - Maximum rounds exceeded

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_btl.py
```

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint
flake8 app/ tests/

# Type checking
mypy app/
```

## Design Principles

- **Production-Lean**: Minimal dependencies, fast startup, efficient resource usage
- **Auditable**: Comprehensive logging and transparency ledger
- **Statistically Rigorous**: Bootstrap confidence intervals and proper convergence testing
- **Modular**: Clean separation of concerns for easy testing and maintenance
- **API-First**: RESTful interface for easy integration

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions, issues, or contributions, please use the GitHub issue tracker.
