from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Face Attendance API")

# Allow requests from your GitHub Pages (simple: allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True}
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import os
import psycopg
from datetime import datetime, timezone, timedelta
from typing import List

app = FastAPI(title="Face Attendance API")

# CORS (keep simple for now; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True}

# ----- DB helper -----
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL env var is missing")

def db():
    # context manager yields a connection
    return psycopg.connect(DB_URL, autocommit=True)

# ----- Models -----
# face-api.js descriptor length is 128 floats; allow 128–256 just in case
class EmbeddingIn(BaseModel):
    embedding: conlist(float, min_items=64, max_items=256)

# ----- Math helpers (no numpy) -----
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)

# =========================================================
# 1) Save/Update a student's face embedding (one-time setup)
#    POST /api/students/{student_id}/embedding
# =========================================================
@app.post("/api/students/{student_id}/embedding")
def save_embedding(student_id: int, body: EmbeddingIn):
    emb = body.embedding
    # optional sanity clamp to 128
    if len(emb) > 128:
        emb = emb[:128]

    with db() as conn:
        with conn.cursor() as cur:
            # ensure student exists and is a 'student'
            cur.execute(
                "SELECT id, role FROM users WHERE id=%s",
                (student_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Student not found")
            if row[1] != "student":
                raise HTTPException(status_code=400, detail="User is not a student")

            # upsert
            cur.execute(
                """
                INSERT INTO student_embeddings (student_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (student_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = NOW()
                """,
                (student_id, emb),
            )
    return {"ok": True, "student_id": student_id, "saved_dims": len(emb)}

# =========================================================
# 2) Check-in with a live embedding & mark attendance
#    POST /api/attendance/checkin-vec?course_id=&session_id=
# =========================================================
@app.post("/api/attendance/checkin-vec")
def checkin_vec(
    body: EmbeddingIn,
    course_id: int = Query(...),
    session_id: int = Query(...),
):
    live = body.embedding
    if len(live) > 128:
        live = live[:128]

    with db() as conn:
        with conn.cursor() as cur:
            # ensure session & course match
            cur.execute(
                """
                SELECT s.id, s.course_id, s.start_time, s.late_after_minutes
                FROM sessions s
                WHERE s.id = %s AND s.course_id = %s
                """,
                (session_id, course_id),
            )
            s = cur.fetchone()
            if not s:
                raise HTTPException(status_code=404, detail="Session not found for course")
            _, _, start_time, late_after_minutes = s

            # fetch enrolled students who have embeddings
            cur.execute(
                """
                SELECT se.student_id, se.embedding
                FROM student_embeddings se
                JOIN enrollments e ON e.student_id = se.student_id
                WHERE e.course_id = %s
                """,
                (course_id,),
            )
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="No embeddings for this course")

            # find best match by cosine similarity
            best_id = None
            best_sim = -1.0
            for sid, emb in rows:
                sim = cosine_similarity(live, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = sid

            # threshold: tune as needed; 0.6–0.7 common for face-api
            THRESHOLD = 0.6
            if best_sim < THRESHOLD or best_id is None:
                raise HTTPException(status_code=404, detail="No matching student")

            # decide present/late
            now = datetime.now(timezone.utc)
            cutoff = start_time + timedelta(minutes=late_after_minutes or 0)
            status = "present" if now <= cutoff else "late"

            # insert or update attendance row
            cur.execute(
                """
                INSERT INTO attendance (session_id, student_id, status, timestamp)
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

