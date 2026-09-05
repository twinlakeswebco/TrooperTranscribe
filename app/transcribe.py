"""
Transcription engine using faster-whisper.
Runs fully offline using locally cached models.
"""

from pathlib import Path
from typing import Callable, Optional
import torch


def transcribe_audio(
    audio_path: str,
    model_name: str,
    models_path: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> list:
    """
    Transcribe an audio/video file using faster-whisper.

    Returns a list of dicts: [{start, end, text}, ...]
    """
    from faster_whisper import WhisperModel

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    
    # FIX: Using 'float16' on some mobile GPUs can cause errors. 
    # 'int8_float16' is the sweet spot for speed and stability on portable hardware.
    compute_type = "int8_float16" if cuda else "int8"
    
    # Must match the per-model download_root used in setup_models.py
    # (models/whisper/<model_name>/...), not just models/whisper/ -
    # otherwise local_files_only=True can't find the cached snapshot.
    whisper_cache = str(models_path / "whisper" / model_name)

    if progress_callback:
        mode_label = "GPU" if cuda else "CPU"
        progress_callback(8, f"Loading {model_name} ({mode_label})...")

    # FIX: Added local_files_only=True to ensure it never tries to use the internet
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=whisper_cache,
        local_files_only=True 
    )

    if progress_callback:
        progress_callback(18, "Analyzing audio waveform...")

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    total_duration = max(info.duration, 1)

    for seg in segments_gen:
        # Filter out empty segments
        if not seg.text.strip():
            continue
            
        segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
        )
        if progress_callback:
            # Scale progress between 18% and 66% (leaving room for diarization)
            pct = 18 + int((seg.end / total_duration) * 48)
            elapsed = int(seg.end)
            total = int(total_duration)
            progress_callback(min(pct, 66), f"Transcribing... {elapsed}s / {total}s")

    return segments