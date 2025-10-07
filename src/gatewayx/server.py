"""
Gateway X Server - FastAPI application
"""

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import time

from .settings import settings
from .secure_config import config
from .orchestrator import Orchestrator
from .engine_pool import EnginePool
from .monitors import SystemMonitor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gateway X",
    description="Multi-Engine AI Consensus System",
    version="32.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine_pool: Optional[EnginePool] = None
orchestrator: Optional[Orchestrator] = None
monitor: Optional[SystemMonitor] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    budget: Optional[int] = None
    confidence_threshold: Optional[float] = None
    engines: Optional[List[str]] = None


class EngineResponse(BaseModel):
    """Individual engine response with scoring"""
    engine: str
    text: str
    cost: float
    tokens: int
    response_time: float
    score: Optional[float] = None
    ranking: Optional[int] = None


class RoundData(BaseModel):
    """Data for a single round of consensus building"""
    round_number: int
    engines_used: List[str]
    responses: List[EngineResponse]
    consensus_data: Optional[Dict[str, Any]] = None
    round_confidence: Optional[float] = None
    threshold_met: bool = False

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    best_claim: str
    confidence: float
    rounds_used: int
    engines_used: List[str]
    total_cost: float
    processing_time: float
    metadata: Dict[str, Any]
    all_responses: List[EngineResponse]
    consensus_data: Optional[Dict[str, Any]] = None
    detailed_rounds: Optional[List[RoundData]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global engine_pool, orchestrator, monitor
    
    logger.info("Starting Gateway X server...")
    
    # Log security configuration
    security_status = config.get_config_summary()
    logger.info(f"🔒 Security Status: {security_status['security']}")
    logger.info(f"🔑 API Keys Status: {security_status['api_keys']}")
    
    try:
        # Initialize engine pool
        engine_pool = EnginePool()
        await engine_pool.initialize()
        
        # Initialize orchestrator
        orchestrator = Orchestrator(engine_pool)
        
        # Initialize system monitor
        monitor = SystemMonitor()
        
        logger.info("Gateway X server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Gateway X server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global engine_pool, orchestrator, monitor
    
    logger.info("Shutting down Gateway X server...")
    
    if engine_pool:
        await engine_pool.cleanup()
    
    if monitor:
        await monitor.cleanup()
    
    logger.info("Gateway X server shutdown complete")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "32.1.0",
        "timestamp": time.time(),
        "engines_available": len(engine_pool.engines) if engine_pool else 0
    }


@app.get("/security/status")
async def security_status():
    """Security status endpoint (safe for logging)"""
    return {
        "status": "secure" if config.is_secure_mode() else "warning",
        "secure_mode": config.is_secure_mode(),
        "local_secrets_loaded": os.path.exists("config/secrets/.env.local"),
        "api_keys_configured": {
            "anthropic": bool(config.get_api_key("anthropic")),
            "openai": bool(config.get_api_key("openai")),
            "google": bool(config.get_api_key("google")),
            "grok": bool(config.get_api_key("grok")),
        },
        "use_real_llm": config.get("use_real_llm", False),
        "timestamp": time.time()
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        start_time = time.time()
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Use settings defaults if not provided
        budget = request.budget or settings.default_budget
        confidence_threshold = request.confidence_threshold or settings.confidence_threshold
        
        logger.info(f"Budget: {budget}, Confidence: {confidence_threshold}")
        
        # Process query
        result = await orchestrator.process_query(
            query=request.query,
            budget=budget,
            confidence_threshold=confidence_threshold,
            engines=request.engines
        )
        
        logger.info(f"Query processed successfully. Result keys: {result.keys()}")
        logger.info(f"Detailed rounds in result: {'detailed_rounds' in result}")
        if 'detailed_rounds' in result:
            logger.info(f"Number of detailed rounds: {len(result['detailed_rounds'])}")
        
        processing_time = time.time() - start_time
        
        # Convert responses to EngineResponse objects with scoring
        all_responses = []
        if "responses" in result:
            # Get BTL scores and rankings from consensus data
            btl_scores = {}
            rankings = {}
            if "consensus_data" in result and result["consensus_data"]:
                btl_scores_list = result["consensus_data"].get("btl_scores", [])
                rankings_list = result["consensus_data"].get("rankings", [])
                
                # Map scores and rankings to responses
                for i, response in enumerate(result["responses"]):
                    if i < len(btl_scores_list):
                        btl_scores[response.get("text", "")] = btl_scores_list[i]
                    if i < len(rankings_list):
                        rankings[response.get("text", "")] = i + 1
            
            for i, response in enumerate(result["responses"]):
                response_text = response.get("text", "")
                
                # Use BTL score if available, otherwise calculate basic score
                if response_text in btl_scores:
                    score = btl_scores[response_text]
                else:
                    # Fallback to length-based scoring
                    score = len(response_text) / 100.0
                
                # Get ranking from consensus data
                ranking = rankings.get(response_text, None)
                
                engine_response = EngineResponse(
                    engine=response.get("engine", "unknown"),
                    text=response_text,
                    cost=response.get("cost", 0.0),
                    tokens=response.get("tokens", 0),
                    response_time=response.get("response_time", 0.0),
                    score=score,
                    ranking=ranking
                )
                all_responses.append(engine_response)
        
        # Convert detailed rounds to proper format
        detailed_rounds = []
        if result.get("detailed_rounds"):
            for round_data in result["detailed_rounds"]:
                # Convert response dictionaries to EngineResponse objects
                engine_responses = []
                for resp in round_data.get("responses", []):
                    engine_response = EngineResponse(
                        engine=resp.get("engine", "unknown"),
                        text=resp.get("text", ""),
                        cost=resp.get("cost", 0.0),
                        tokens=resp.get("tokens", 0),
                        response_time=resp.get("response_time", 0.0),
                        score=resp.get("score"),
                        ranking=resp.get("ranking")
                    )
                    engine_responses.append(engine_response)
                
                round_obj = RoundData(
                    round_number=round_data.get("round_number", 1),
                    engines_used=round_data.get("engines_used", []),
                    responses=engine_responses,
                    consensus_data=round_data.get("consensus_data"),
                    round_confidence=round_data.get("round_confidence"),
                    threshold_met=round_data.get("threshold_met", False)
                )
                detailed_rounds.append(round_obj)
        
        return QueryResponse(
            best_claim=result["best_claim"],
            confidence=result["confidence"],
            rounds_used=result["rounds_used"],
            engines_used=result["engines_used"],
            total_cost=result.get("total_cost", 0.0),
            processing_time=processing_time,
            metadata=result.get("metadata", {}),
            all_responses=all_responses,
            consensus_data=result.get("consensus_data"),
            detailed_rounds=detailed_rounds
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines/status")
async def get_engine_status():
    """Get status of all engines"""
    if not engine_pool:
        raise HTTPException(status_code=503, detail="Engine pool not initialized")
    
    return {
        "engines": engine_pool.get_status(),
        "total_engines": len(engine_pool.engines),
        "available_engines": len([e for e in engine_pool.engines.values() if e.is_available])
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    return await monitor.get_metrics()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gateway X - Multi-Engine AI Consensus System",
        "version": "32.1.0",
        "docs": "/docs",
        "health": "/health"
    }


def main():
    """Main entry point for running the server"""
    import uvicorn
    
    uvicorn.run(
        "gatewayx.server:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        reload=True
    )


if __name__ == "__main__":
    main()
