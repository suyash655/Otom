"""
OtoScope AI — FastAPI application entry point.

Only contains:
  - App factory
  - Lifespan context (startup/shutdown hooks)
  - Middleware registration
  - Router inclusion

No business logic lives here.

Run locally:
  uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-warm the model so the first request isn't slow."""
    try:
        from backend.app.core.dependencies import get_model
        get_model()
    except Exception:
        pass  # Startup failure is non-fatal; endpoints will surface the error
    yield
    # Shutdown hooks can be added here


app = FastAPI(
    title="OtoScope AI API",
    description=(
        "Hybrid CNN+TDA model for otoscopic image classification "
        "with explainability (Grad-CAM, TDA importance, clinical reasoning).\n\n"
        "**INTENDED USE:** This is an unvalidated research prototype developed "
        "for academic exploration of hybrid CNN+TDA architectures. It is NOT a "
        "medical device and has NOT been clinically validated. It must not be used "
        "for diagnosis, treatment decisions, or patient care. All outputs require "
        "review by a qualified clinician."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
