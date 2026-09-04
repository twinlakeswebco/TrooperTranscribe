import os
import uuid
from pathlib import Path
from typing import Optional
import torch

# --- EAGER IMPORTS TO PREVENT RECURSION ERROR ---
# We manually trigger these imports to stop speechbrain's lazy-loader from looping
try:
    import speechbrain
    import speechbrain.utils.importutils
except ImportError:
    pass
# -----------------------------------------------

def diarize_audio(audio_path: str, models_path: Path, num_speakers: Optional[int] = None) -> list:
    from pyannote.audio import Pipeline
    import av
    import tempfile
    import numpy as np
    import soundfile as sf

    result = []
    
    # 1. Standardize Audio (16kHz Mono)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()

    container = av.open(audio_path)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    samples = []
    for frame in container.decode(audio=0):
        for reframe in resampler.resample(frame):
            samples.append(reframe.to_ndarray()[0])
    container.close()

    audio_array = np.concatenate(samples).astype(np.float32)
    sf.write(tmp_wav.name, audio_array, 16000)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pipeline. Fully offline (HF_HUB_OFFLINE=1, set in main.py) -
        # resolves entirely from the local cache built by setup_models.py, so
        # no auth token is needed here.
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if pipeline is None:
            raise RuntimeError(
                "Could not load the pyannote/speaker-diarization-3.1 pipeline "
                "from the local model cache. Re-run setup_models.py on the "
                "build machine and repackage the 'models' folder."
            )
        pipeline.to(device)

        audio_data, sr = sf.read(tmp_wav.name, dtype="float32")
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        # 2. Run Diarization
        diarization = pipeline(audio_input, **kwargs)

        # pyannote 3.x returns Annotation directly
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append({
                "start": round(turn.start, 3), 
                "end": round(turn.end, 3), 
                "speaker": speaker
            })
            
        return result

    finally:
        if os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)

def merge_transcript_diarization(segments: list, diarization: list) -> list:
    """
    Combines the text segments from Whisper with the speaker labels from Pyannote.
    """
    speaker_map = {}
    counter = [1]
    
    def normalize(raw: str) -> str:
        if raw not in speaker_map:
            speaker_map[raw] = f"SPEAKER_{counter[0]:02d}"
            counter[0] += 1
        return speaker_map[raw]

    result = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker_raw, best_overlap = None, 0.0
        
        for d in diarization:
            overlap = max(0.0, min(seg_end, d["end"]) - max(seg_start, d["start"]))
            if overlap > best_overlap:
                best_overlap, best_speaker_raw = overlap, d["speaker"]
        
        assigned = normalize(best_speaker_raw) if best_speaker_raw else "SPEAKER_01"
        result.append({
            "id": str(uuid.uuid4()), 
            "start": seg_start, 
            "end": seg_end, 
            "text": seg["text"], 
            "speaker": assigned
        })
    return result
