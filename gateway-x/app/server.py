"""FastAPI server for Gateway X consensus engine."""

import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import CONFIG
from .schemas import QueryIn, QueryOut, TimelineItem
from .orchestrator import Orchestrator
from .report import render_timeline_html


logger = logging.getLogger("gatewayx")

# Initialize FastAPI app
app = FastAPI(title="Gateway X", version="30.2", description="Production-lean consensus engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize orchestrator
ORCH = Orchestrator(CONFIG)


@app.post("/query", response_model=QueryOut)
async def handle_query(body: QueryIn):
    """Handle consensus query request."""
    try:
        res = await ORCH.run(body.query, body.budget, body.confidence_threshold)
        return QueryOut(**res)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/timeline/{run_id}")
async def get_timeline(run_id: str):
    """Get timeline for a specific run as JSON."""
    try:
        items = ORCH.get_timeline(run_id)
        return [it.model_dump() for it in items]
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/timeline/{run_id}/report", response_class=HTMLResponse)
async def get_timeline_report(run_id: str):
    """Get timeline report as HTML."""
    try:
        items = ORCH.get_timeline(run_id)
        html_doc = render_timeline_html(run_id, items)
        return HTMLResponse(content=html_doc, status_code=200)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend interface."""
    frontend_path = Path(__file__).parent.parent / "static" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    else:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Gateway X API</h1>
                <p>Frontend not found. API endpoints available at:</p>
                <ul>
                    <li>POST /query - Submit consensus queries</li>
                    <li>GET /timeline/{run_id} - Get timeline data</li>
                    <li>GET /timeline/{run_id}/report - Get HTML report</li>
                    <li>GET /health - Health check</li>
                </ul>
            </body>
        </html>
        """)

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "service": "Gateway X", 
        "version": "30.2", 
        "use_real_llm": CONFIG.USE_REAL_LLM
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Gateway X"}


@app.get("/test-engines")
async def test_engines():
    """Test all AI engines to ensure they're working."""
    from .ai_engines import MultiEngineClient
    
    try:
        client = MultiEngineClient(CONFIG)
        results = await client.test_all_engines()
        
        return {
            "engines": results,
            "all_working": all(results.values()),
            "total_engines": len(results),
            "working_engines": sum(results.values())
        }
    except Exception as e:
        logger.exception("Error testing engines: %s", e)
        return {
            "error": str(e),
            "engines": {},
            "all_working": False
        }
