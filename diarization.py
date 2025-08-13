import argparse
import os
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Whisper
import whisper

# pyannote
from pyannote.audio import Pipeline

import warnings

warnings.filterwarnings("ignore")
# --------------------------- Utilities ---------------------------


def ffmpeg_extract_segment(
    src_path: str,
    dst_path: str,
    start_time: float,
    end_time: float,
    mono: bool = True,
    sample_rate: int = 16000,
) -> None:
    """
    Extract [start_time, end_time) segment using ffmpeg, re-encode to mono WAV 16k.
    """
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        # create a tiny silent file to avoid crashes; will be ignored by caller
        duration = 0.01
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        src_path,
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-vn",
        "-sn",
        "-loglevel",
        "error",
        dst_path,
    ]
    subprocess.run(cmd, check=True)


def seconds_round(x: float, ndigits: int = 3) -> float:
    return float(f"{x:.{ndigits}f}")


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class LabeledText:
    start: float
    end: float
    speaker: str
    text: str


def split_into_chunks(
    start: float, end: float, max_len: float
) -> List[Tuple[float, float]]:
    """Split [start, end) into sub-intervals of at most max_len seconds."""
    chunks = []
    t = start
    while t < end - 1e-6:
        t2 = min(end, t + max_len)
        chunks.append((t, t2))
        t = t2
    return chunks


# --------------------------- Core Pipeline ---------------------------


def run_diarization(audio_path: str, hf_token: Optional[str]) -> List[SpeakerTurn]:
    """
    Run pyannote speaker diarization on the whole file.
    """
    if not hf_token:
        hf_token = "YOUR_HF_TOKEN"
    if not hf_token:
        raise RuntimeError(
            "A Hugging Face token is required. Pass --hf_token or set HUGGINGFACE_TOKEN."
        )

    # Recommended pipeline; model requires HF token access
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="YOUR_HF_TOKEN",
    )

    diarization = pipeline("./audio.wav")
    turns: List[SpeakerTurn] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=seconds_round(segment.start),
                end=seconds_round(segment.end),
                speaker=speaker,
            )
        )
    # Sort chronologically
    turns.sort(key=lambda x: (x.start, x.end))
    return turns


def run_whisper_on_segment(
    model,
    audio_path: str,
    seg_start: float,
    seg_end: float,
    language: Optional[str],
    device: Optional[str],
    tmp_dir: str,
) -> str:
    """
    Extract segment and transcribe with whisper.
    """
    seg_wav = os.path.join(tmp_dir, f"seg_{seg_start:.3f}_{seg_end:.3f}.wav")
    ffmpeg_extract_segment(audio_path, seg_wav, seg_start, seg_end)

    # Whisper: you can set fp16=False for CPU-only inference
    result = model.transcribe(
        seg_wav, language=language, fp16=False if device == "cpu" else None
    )
    return result.get("text", "").strip()


def transcribe_with_diarization(
    audio_path: str,
    whisper_model_size: str = "small",
    language: Optional[str] = None,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    max_chunk: float = 29.5,
    merge_gap: float = 1.0,
) -> List[LabeledText]:
    """
    Full pipeline:
    - Diarize whole audio.
    - For each diarized turn, split into <=30s chunks for Whisper, transcribe, and stitch.
    - Merge adjacent turns of same speaker separated by < merge_gap seconds.
    """
    # Load Whisper
    model = whisper.load_model(whisper_model_size, device=device if device else None)

    # Diarize
    turns = run_diarization("./audio.m4a", hf_token)

    labeled: List[LabeledText] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for t in turns:
            # Split each speaker turn into <= max_chunk slices
            subchunks = split_into_chunks(t.start, t.end, max_chunk)
            pieces = []
            for s0, s1 in subchunks:
                txt = run_whisper_on_segment(
                    model, audio_path, s0, s1, language, device, tmpdir
                )
                if txt:
                    pieces.append(txt)
            merged_text = " ".join(pieces).strip()
            if merged_text:
                labeled.append(
                    LabeledText(
                        start=t.start, end=t.end, speaker=t.speaker, text=merged_text
                    )
                )

    # Merge consecutive same-speaker entries with small gap
    merged: List[LabeledText] = []
    for lt in labeled:
        if not merged:
            merged.append(lt)
            continue
        prev = merged[-1]
        if lt.speaker == prev.speaker and (lt.start - prev.end) <= merge_gap:
            # merge
            merged[-1] = LabeledText(
                start=prev.start,
                end=lt.end,
                speaker=prev.speaker,
                text=(prev.text + " " + lt.text).strip(),
            )
        else:
            merged.append(lt)

    return merged


# --------------------------- CLI ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Diarized Whisper ASR with 30s chunks."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio/video file (any ffmpeg-readable format).",
    )
    parser.add_argument(
        "--whisper_model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language (e.g., 'en', 'hi'). Let Whisper auto-detect if None.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token (or set HUGGINGFACE_TOKEN env var).",
    )
    parser.add_argument(
        "--device", default=None, help="Force device for Whisper (e.g., 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--max_chunk_sec",
        type=float,
        default=29.5,
        help="Max chunk length per Whisper pass (<=30s recommended).",
    )
    parser.add_argument(
        "--merge_gap_sec",
        type=float,
        default=1.0,
        help="Merge turns of same speaker if gap ≤ this (seconds).",
    )
    args = parser.parse_args()

    results = transcribe_with_diarization(
        audio_path=args.audio,
        whisper_model_size=args.whisper_model,
        language=args.language,
        hf_token=args.hf_token,
        device=args.device,
        max_chunk=args.max_chunk_sec,
        merge_gap=args.merge_gap_sec,
    )

    # Print in requested format
    for r in results:
        # Normalize speaker label to "speakerX" format if desired:
        # pyannote uses "SPEAKER_00", "SPEAKER_01" — keep as-is or map here.
        print(f"{r.speaker.lower()}: {r.text}")


if __name__ == "__main__":
    main()
