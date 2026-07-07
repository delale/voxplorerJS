"""
Voxplorer FastAPI Backend Application

Main entry point for the Voxplorer API. Handles feature extraction,
dimensionality reduction, and data processing.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import data, features, reduction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Voxplorer API starting up...")
    yield
    logger.info("Voxplorer API shutting down...")


# Initialize FastAPI application
app = FastAPI(
    title="Voxplorer API",
    description="Backend API for interactive voice data visualization and analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React dev server
        "http://localhost:5000",  # API server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware that returns JSON responses."""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
            },
        )


# Register routers
app.include_router(data.router)
app.include_router(features.router)
app.include_router(reduction.router)


# Root API endpoint
@app.get("/api")
async def api_root() -> dict[str, Any]:
    """
    Root API endpoint providing API information.

    Returns:
        dict: API metadata including title, version, and available routes
    """
    return {
        "title": "Voxplorer API",
        "version": "0.1.0",
        "description": "Backend API for interactive voice data visualization and analysis",
        "endpoints": {
            "health": "/api/health",
            "features": "/api/features",
            "reduction": "/api/reduction",
            "data": "/api/data",
        },
    }


# Health check endpoint
@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint to verify API is running.

    Returns:
        dict: Status of the API
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
