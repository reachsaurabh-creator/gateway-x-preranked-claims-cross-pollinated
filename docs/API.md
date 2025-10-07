# Gateway X API Documentation

## Overview

Gateway X provides a RESTful API for multi-engine AI consensus building. The API allows you to submit queries and receive consensus responses from multiple AI models.

## Base URL

```
http://localhost:3001
```

## Authentication

Currently, no authentication is required. API keys are configured server-side.

## Endpoints

### Health Check

Check if the service is running.

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "32.1.0",
  "timestamp": 1640995200.0,
  "engines_available": 3
}
```

### Query

Submit a query for consensus processing.

**POST** `/query`

**Request Body:**
```json
{
  "query": "What is artificial intelligence?",
  "budget": 20,
  "confidence_threshold": 0.95,
  "engines": ["anthropic", "openai", "google"]
}
```

**Parameters:**
- `query` (string, required): The question to ask
- `budget` (integer, optional): Maximum cost budget (default: 20)
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.95)
- `engines` (array, optional): Specific engines to use

**Response:**
```json
{
  "best_claim": "Artificial intelligence (AI) is a branch of computer science...",
  "confidence": 0.97,
  "rounds_used": 3,
  "engines_used": ["anthropic", "openai", "google"],
  "total_cost": 0.045,
  "processing_time": 2.3,
  "metadata": {
    "budget_used": 20,
    "confidence_threshold": 0.95,
    "multi_engine_mode": true,
    "consensus_enabled": true
  }
}
```

### Engine Status

Get status of all available engines.

**GET** `/engines/status`

**Response:**
```json
{
  "engines": {
    "anthropic": {
      "name": "anthropic",
      "is_available": true,
      "total_requests": 150,
      "total_tokens": 15000,
      "total_cost": 3.75,
      "error_count": 2,
      "last_request_time": 1640995200.0
    },
    "openai": {
      "name": "openai",
      "is_available": true,
      "total_requests": 120,
      "total_tokens": 12000,
      "total_cost": 6.0,
      "error_count": 1,
      "last_request_time": 1640995190.0
    }
  },
  "total_engines": 2,
  "available_engines": 2
}
```

### Metrics

Get system metrics and statistics.

**GET** `/metrics`

**Response:**
```json
{
  "uptime": 3600.0,
  "total_metrics": 500,
  "recent_metrics": 50,
  "health_checks": {
    "engine_pool": {
      "status": "healthy",
      "last_check": 1640995200.0,
      "result": true
    }
  },
  "metrics_summary": {
    "query_processing_time": {
      "count": 100,
      "average": 2.3,
      "min": 0.5,
      "max": 5.2,
      "latest": 2.1
    }
  },
  "alerts": []
}
```

## Error Responses

All errors return a JSON response with error details:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error
- `503`: Service Unavailable (engines not ready)

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## Examples

### Basic Query

```bash
curl -X POST "http://localhost:3001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "budget": 10,
    "confidence_threshold": 0.9
  }'
```

### Multi-Engine Query

```bash
curl -X POST "http://localhost:3001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing",
    "budget": 50,
    "confidence_threshold": 0.95,
    "engines": ["anthropic", "openai", "google"]
  }'
```

### Check Engine Status

```bash
curl "http://localhost:3001/engines/status"
```

## Configuration

The API behavior can be configured using environment variables:

- `GATEWAYX_MULTI_ENGINE_MODE`: Enable multi-engine mode
- `GATEWAYX_ENABLE_CONSENSUS_JUDGING`: Enable consensus judging
- `GATEWAYX_DEFAULT_BUDGET`: Default budget for queries
- `GATEWAYX_CONFIDENCE_THRESHOLD`: Default confidence threshold
- `GATEWAYX_LOAD_BALANCING_STRATEGY`: Load balancing strategy (weighted, round_robin, least_loaded)

See the main README for a complete list of configuration options.
