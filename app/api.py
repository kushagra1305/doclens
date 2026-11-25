"""
app/api.py - Step 9 of DocLens: FastAPI backend.

Run (from the doclens/ folder):
    uvicorn app.api:app --reload

Endpoints:
    GET  /health  -> {"status": "ok", ...}
    POST /ask     -> {"question": "...", "model": "qwen2.5:7b"}   (model is optional)
Interactive docs: http://localhost:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import generate  # noqa: E402
from retrieve import search  # noqa: E402

SNIPPET_CHARS = 1200


@asynccontextmanager
async def lifespan(app: FastAPI):
    search("warm up", 1, "vecrerank")  # load the models once, at startup
    yield


app = FastAPI(title="DocLens", description="Q&A over tax documents with citations", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "default_model": generate.MODEL, "rerank_cutoff": generate.RERANK_CUTOFF}


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    result = generate.answer(question, req.model)

    cited = []
    for s in result["sources"]:
        for n in s["passages"]:
            hit = result["hits"][n - 1]
            cited.append(
                {
                    "n": n,
                    "source": hit["source"],
                    "page": hit["page"],
                    "text": hit["text"][:SNIPPET_CHARS],
                }
            )

    return {
        "question": question,
        "answer": result["answer"],
        "refused_by": result["refused_by"],
        "cited_by_model": result["cited"],
        "top_score": result["top_score"],
        "sources": result["sources"],
        "passages": cited,
        "model": req.model or generate.MODEL,
    }