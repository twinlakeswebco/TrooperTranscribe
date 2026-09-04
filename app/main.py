# Copyright (c) 2026 Casey Keown. All rights reserved. Proprietary and confidential.

"""
Trooper Transcribe™
Portable offline transcription tool.
All processing is local. No data leaves this machine.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

# Resolve models path first
_default_models = Path(__file__).parent.parent / "models"
MODELS_PATH = Path(os.environ.get("KSP_MODELS_PATH", str(_default_models)))

# Set HuggingFace environment BEFORE any HF imports
# Offline flags are set by launch.bat and must not be overridden here -
# forcing them online breaks pyannote's gated-model auth check and makes
# this process reach the network, which contradicts "no data leaves this
# machine". Only fill them in if something ran this outside launch.bat.
os.environ["HF_HOME"] = str(MODELS_PATH / "pyannote")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import torch

app = FastAPI(title="TrooperTranscribe™", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store
jobs: dict = {}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/hardware")
async def hardware_info():
    cuda = torch.cuda.is_available()
    if cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA GPU"
        return {"mode": "GPU", "device": gpu_name, "cuda": True}
    return {"mode": "CPU", "device": "CPU — int8 optimized", "cuda": False}


@app.post("/api/transcribe")
async def start_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = "large-v2",
    num_speakers: str = "auto",
):
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix or ".tmp"

    # Save uploaded file to a temporary location
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Queued...",
        "result": None,
        "error": None,
        "filename": file.filename,
    }

    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        audio_path=tmp.name,
        model_name=model,
        num_speakers=num_speakers,
    )

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


class ExportRequest(BaseModel):
    segments: list
    filename: str = "transcript"


@app.post("/api/export/{fmt}")
async def export_transcript(fmt: str, req: ExportRequest):
    from app.export import export_transcript as do_export
    try:
        data, mime, ext = do_export(req.segments, fmt, req.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    safe_name = req.filename.replace('"', "")
    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.{ext}"'},
    )


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _set_progress(job_id: str, progress: int, message: str):
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"] = message


def run_pipeline(job_id: str, audio_path: str, model_name: str, num_speakers: str):
    try:
        from app.transcribe import transcribe_audio
        from app.diarize import diarize_audio, merge_transcript_diarization

        _set_progress(job_id, 5, "Loading transcription model...")

        # 1. Whisper Transcription
        segments = transcribe_audio(
            audio_path=audio_path,
            model_name=model_name,
            models_path=MODELS_PATH,
            progress_callback=lambda p, m: _set_progress(job_id, p, m),
        )

        _set_progress(job_id, 70, "Running speaker diarization...")

        # 2. Pyannote Diarization
        n_speakers = None if num_speakers == "auto" else int(num_speakers)
        diarization = diarize_audio(
            audio_path=audio_path,
            models_path=MODELS_PATH,
            num_speakers=n_speakers,
        )

        _set_progress(job_id, 90, "Merging transcript with speakers...")

        # 3. Merging
        result = merge_transcript_diarization(segments, diarization)

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Complete"
        jobs[job_id]["result"] = result

    except Exception as exc:
        import traceback
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["detail"] = traceback.format_exc()
        print(f"Error in job {job_id}: {traceback.format_exc()}")

    finally:
        # Cleanup temporary audio file
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except OSError:
            pass
