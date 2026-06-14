from __future__ import annotations

import os
import sys

# Ajoute src/ au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quorex.core.memory.manager import MemoryManager, MemoryConfig

# ─── Global instance ─────────────────────────────────────────────────────────

manager: MemoryManager | None = None

SEED = [
    {"action": "viewed pricing page",  "metadata": {"text": "viewed pricing"}},
    {"action": "searched pricing",     "metadata": {"text": "searched pricing"}},
    {"action": "upgraded plan pro",    "metadata": {"text": "upgraded to pro"}},
    {"action": "visited homepage",     "metadata": {"text": "visited homepage"}},
    {"action": "searched docs api",    "metadata": {"text": "searched docs"}},
    {"action": "clicked cta button",   "metadata": {"text": "clicked cta"}},
    {"action": "write code python",    "metadata": {"text": "coding python"}},
    {"action": "read documentation",   "metadata": {"text": "reading docs"}},
    {"action": "created api key",      "metadata": {"text": "created api key"}},
    {"action": "sent message chat",    "metadata": {"text": "sent message"}},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    config = MemoryConfig(
        db_path      = os.getenv("QUOREX_DB_PATH",      "/data/quorex_memory"),
        encoder_path = os.getenv("QUOREX_ENCODER_PATH", "/data/quorex_encoder"),
        n_components = int(os.getenv("QUOREX_DIMS", "32")),
        top_k        = 5,
        threshold    = 0.05,
    )
    manager = MemoryManager(config)
    manager.start(seed_events=SEED)
    yield
    manager.stop()

app = FastAPI(title="Quorex Memory Engine", version="0.1.0", lifespan=lifespan)

# ─── Auth ─────────────────────────────────────────────────────────────────────

INTERNAL_KEY = os.getenv("QUOREX_INTERNAL_KEY", "")

def verify_key(x_quorex_internal_key: str = Header(...)):
    if not INTERNAL_KEY:
        raise HTTPException(status_code=500, detail="QUOREX_INTERNAL_KEY not set")
    if x_quorex_internal_key != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="Invalid internal key")

# ─── Schemas ──────────────────────────────────────────────────────────────────

class StoreRequest(BaseModel):
    user_id   : str
    action    : str
    text      : str
    metadata  : dict = {}
    timestamp : float | None = None

class RetrieveRequest(BaseModel):
    user_id   : str
    query     : str
    top_k     : int | None = None
    threshold : float | None = None
    # Optional reference time for decay. When the caller "freezes" a user,
    # it pins `now` so temporal decay stops advancing. Defaults to current time.
    now       : float | None = None

class ForgetRequest(BaseModel):
    user_id : str
    vec_id  : int

class PurgeRequest(BaseModel):
    user_id : str

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if manager is None or not manager._started:
        return JSONResponse({"status": "starting"}, status_code=503)
    stats = manager.stats()
    return {"status": "ok", "total_vectors": stats.get("total_vectors", 0)}


@app.post("/memory/store", dependencies=[Depends(verify_key)])
def store(req: StoreRequest):
    try:
        event = {
            "action": req.action,
            "metadata": {"text": req.text, **req.metadata}
        }
        vec_id = manager.remember(req.user_id, event, timestamp=req.timestamp)
        return {"vec_id": vec_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/retrieve", dependencies=[Depends(verify_key)])
def retrieve(req: RetrieveRequest):
    try:
        memories = manager.recall(
            req.user_id,
            req.query,
            top_k=req.top_k,
            threshold=req.threshold,
            now=req.now,
        )
        return {
            "memories": [
                {
                    "vec_id":         m.vec_id,
                    "text":           m.text,
                    "action":         m.action,
                    "timestamp":      m.timestamp,
                    "final_score":    m.final_score,
                    "cosine_sim":     m.cosine_sim,
                    "decay_weight":   m.decay_weight,
                    "freq_weight":    m.freq_weight,
                    "hours_ago":      m.hours_ago,
                    "reinforcements": m.reinforcements,
                    "meta":           m.meta,
                }
                for m in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/forget", dependencies=[Depends(verify_key)])
def forget(req: ForgetRequest):
    try:
        deleted = manager.forget(req.user_id, req.vec_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/purge", dependencies=[Depends(verify_key)])
def purge(req: PurgeRequest):
    try:
        count = manager.purge(req.user_id)
        if count == 0:
            raise HTTPException(status_code=404, detail="User not found or no memories")
        return {"deleted": count}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats", dependencies=[Depends(verify_key)])
def stats(user_id: str | None = None):
    try:
        return manager.stats(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/export", dependencies=[Depends(verify_key)])
def export(user_id: str):
    """
    Returns ALL live memories for a user (GDPR data portability).

    Read-only: enumerates the segment metadata directly without touching the
    MemoryManager / scorer / index. Soft-deleted vectors are already removed
    from `_metadata`, so only live memories are returned.
    """
    try:
        seg = manager.engine.segment
        user_meta = seg._metadata.get(user_id, {})
        memories = []
        for vec_id, meta in user_meta.items():
            inner = meta.get("metadata", {}) or {}
            memories.append({
                "vec_id":         vec_id,
                "action":         meta.get("action", ""),
                "text":           inner.get("text", meta.get("action", "")),
                "timestamp":      inner.get("timestamp"),
                "reinforcements": inner.get("reinforcements", 1),
                "meta":           meta,
            })
        memories.sort(key=lambda m: (m["timestamp"] or 0))
        return {"user_id": user_id, "count": len(memories), "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))