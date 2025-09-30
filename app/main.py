from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import os
import psycopg
from datetime import datetime, timezone, timedelta
from typing import List

app = FastAPI(title="Face Attendance API")

# CORS: keep * for demo; later lock to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True}

# ---- DB ----
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL env var is missing")

def db():
    return psycopg.connect(DB_URL, autocommit=True)

# ---- Models ----
class EmbeddingIn(BaseModel):
    # face-api.js descriptor ~128 floats
    embedding: conlist(float, min_items=64, max_items=256)

# ---- Math (no numpy needed) ----
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)

# ===== 1) Save/Update a student's face embedding =====
@app.post("/api/students/{student_id}/embedding")
def save_embedding(student_id: int, body: EmbeddingIn):
    emb = body.embedding[:128]  # clamp if longer
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT role FROM public.users WHERE id=%s", (student_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Student not found")
        if row[0] != "student":
            raise HTTPException(status_code=400, detail="User is not a student")

        cur.execute("""
            INSERT INTO public.student_embeddings (student_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (student_id)
            DO UPDATE SET embedding = EXCLUDED.embedding, created_at = NOW()
        """, (student_id, emb))
    return {"ok": True, "student_id": student_id, "saved_dims": len(emb)}

# ===== 2) Match embedding & mark attendance =====
@app.post("/api/attendance/checkin-vec")
def checkin_vec(
    body: EmbeddingIn,
    course_id: int = Query(...),
    session_id: int = Query(...),
):
    live = body.embedding[:128]
    with db() as conn, conn.cursor() as cur:
        # validate session belongs to course
        cur.execute("""
            SELECT start_time, late_after_minutes
            FROM public.sessions
            WHERE id=%s AND course_id=%s
        """, (session_id, course_id))
        s = cur.fetchone()
        if not s:
            raise HTTPException(status_code=404, detail="Session not found for course")
        start_time, late_after = s

        # get enrolled students with embeddings
        cur.execute("""
            SELECT se.student_id, se.embedding
            FROM public.student_embeddings se
            JOIN public.enrollments e ON e.student_id = se.student_id
            WHERE e.course_id = %s
        """, (course_id,))
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No embeddings for this course")

        # best cosine match
        best_id, best_sim = None, -1.0
        for sid, emb in rows:
            sim = cosine_similarity(live, emb)
            if sim > best_sim:
                best_id, best_sim = sid, sim

        if best_id is None or best_sim < 0.60:  # tune threshold as needed
            raise HTTPException(status_code=404, detail="No matching student")

        # present/late
        now = datetime.now(timezone.utc)
        cutoff = start_time + timedelta(minutes=late_after or 0)
        status = "present" if now <= cutoff else "late"

        # upsert attendance
        cur.execute("""
            INSERT INTO public.attendance (session_id, student_id, status, timestamp)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (session_id, student_id)
            DO UPDATE SET status = EXCLUDED.status, timestamp = NOW()
        """, (session_id, best_id, status))

    return {"ok": True, "matched_student_id": best_id, "similarity": round(best_sim, 4),
            "status": status, "course_id": course_id, "session_id": session_id}


