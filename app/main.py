import json
from psycopg.types.json import Json   # <— NEW (psycopg3 helper for JSON)

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import psycopg
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Face Attendance API")

from fastapi.middleware.cors import CORSMiddleware

# --- CORS: allow any origin, no credentials (safest to get you unblocked) ---
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",         # allow ALL origins
    allow_credentials=False,         # must be False when using "*" / regex
    allow_methods=["*"],             # allow all HTTP methods
    allow_headers=["*"],             # allow all headers
    expose_headers=["*"],            # not required, but harmless
    max_age=86400,                   # cache preflight 24h
)

# (Optional) handle any stray OPTIONS requests early
@app.options("/{rest_of_path:path}")
def preflight(rest_of_path: str):
    return Response(status_code=204)

@app.get("/")
def health():
    return {"ok": True}

# ---- DB helper (lazy; won't crash server at import) ----
def get_db_url() -> str:
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise HTTPException(status_code=500, detail="DB_URL environment variable is not set")
    return db_url

def db():
    return psycopg.connect(get_db_url(), autocommit=True)

# ---- Models (keep it simple for OpenAPI & client) ----
class EmbeddingIn(BaseModel):
    embedding: List[float]

def _normalize_pg_url(url: str) -> str:
    # Render/other hosts sometimes hand out "postgres://"
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Ensure SSL (safe on most hosted PGs; harmless if already present)
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url

def get_db_url() -> str:
    url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        # keep the message explicit so it's obvious in logs
        raise HTTPException(status_code=500, detail="Set DATABASE_URL (or DB_URL)")
    return _normalize_pg_url(url)

def db():
    # psycopg 3.x
    return psycopg.connect(get_db_url(), autocommit=True)

# ===== 1) Save/Update a student's face embedding (stored as JSONB) =====
@app.post("/api/students/{student_id}/embedding")
def save_embedding(student_id: int, body: EmbeddingIn):
    emb = body.embedding
    if not isinstance(emb, list) or len(emb) < 64:
        raise HTTPException(status_code=400, detail="Invalid embedding length")
    # keep at most 128 dims (face-api.js descriptor length)
    emb = emb[:128]

    with db() as conn, conn.cursor() as cur:
        # validate user exists and is a student
        cur.execute("SELECT role FROM public.users WHERE id=%s", (student_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Student not found")
        if row[0] != "student":
            raise HTTPException(status_code=400, detail="User is not a student")

        # store as JSONB using psycopg3's Json wrapper
        cur.execute(
            """
            INSERT INTO public.student_embeddings (student_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (student_id)
            DO UPDATE SET embedding = EXCLUDED.embedding, created_at = NOW()
            """,
            (student_id, Json(emb)),
        )

    return {"ok": True, "student_id": student_id, "saved_dims": len(emb)}


# ===== 2) Match embedding & mark attendance =====
@app.post("/api/attendance/checkin-vec")
def checkin_vec(
    body: EmbeddingIn,
    course_id: int = Query(...),
    session_id: int = Query(...),
):
    live = body.embedding
    if not isinstance(live, list) or len(live) < 64:
        raise HTTPException(status_code=400, detail="Invalid embedding length")
    live = live[:128]

    with db() as conn, conn.cursor() as cur:
        # make sure this session belongs to the course
        cur.execute(
            """
            SELECT start_time, late_after_minutes
            FROM public.sessions
            WHERE id=%s AND course_id=%s
            """,
            (session_id, course_id),
        )
        s = cur.fetchone()
        if not s:
            raise HTTPException(status_code=404, detail="Session not found for course")
        start_time, late_after = s

        # get embeddings for students enrolled in this course
        cur.execute(
            """
            SELECT se.student_id, se.embedding
            FROM public.student_embeddings se
            JOIN public.enrollments e ON e.student_id = se.student_id
            WHERE e.course_id = %s
            """,
            (course_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No embeddings for this course")

        # psycopg3 will usually decode jsonb to Python lists automatically.
        # But be defensive in case the driver returns a string.
        best_id, best_sim = None, -1.0
        for sid, emb_obj in rows:
            if isinstance(emb_obj, str):
                try:
                    emb = json.loads(emb_obj)
                except Exception:
                    continue
            else:
                emb = emb_obj  # likely already a list from jsonb

            sim = cosine_similarity(live, emb)
            if sim > best_sim:
                best_id, best_sim = sid, sim

        # require a decent match
        if best_id is None or best_sim < 0.60:
            raise HTTPException(status_code=404, detail="No matching student")

        # present/late
        now = datetime.now(timezone.utc)
        cutoff = start_time + timedelta(minutes=(late_after or 0))
        status = "present" if now <= cutoff else "late"

        # upsert attendance (UNIQUE(session_id, student_id) assumed)
        cur.execute(
            """
            INSERT INTO public.attendance (session_id, student_id, status, timestamp)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (session_id, student_id)
            DO UPDATE SET status = EXCLUDED.status, timestamp = NOW()
            """,
            (session_id, best_id, status),
        )

    return {
        "ok": True,
        "matched_student_id": best_id,
        "similarity": round(best_sim, 4),
        "status": status,
        "course_id": course_id,
        "session_id": session_id,
    }


