"""Frozen text-tokenizer contract used by SemBridge training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SPEECH_TASK_TOKEN = "<|speech|>"
SINGING_TASK_TOKEN = "<|singing|>"
EXPECTED_TEXT_VOCAB_SIZE = 74_080
EXPECTED_SPEECH_TASK_ID = 74_078
EXPECTED_SINGING_TASK_ID = 74_079


def svs_special_tokens() -> list[str]:
    return (
        [f"<|note_pitch_{index}|>" for index in range(129)]
        + [f"<|note_dur_{index}|>" for index in range(500)]
        + ["<SP>", SPEECH_TASK_TOKEN, SINGING_TASK_TOKEN]
    )


def _base_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def extend_svs_vocabulary(tokenizer: Any) -> Any:
    """Add the 632 SVS symbols in the historical order and verify fixed IDs."""

    tokenizer.add_special_tokens({"additional_special_tokens": svs_special_tokens()})
    speech_id = int(tokenizer.convert_tokens_to_ids(SPEECH_TASK_TOKEN))
    singing_id = int(tokenizer.convert_tokens_to_ids(SINGING_TASK_TOKEN))
    if len(tokenizer) != EXPECTED_TEXT_VOCAB_SIZE:
        raise ValueError(
            f"SemBridge tokenizer must contain {EXPECTED_TEXT_VOCAB_SIZE} tokens, "
            f"got {len(tokenizer)}"
        )
    if (speech_id, singing_id) != (
        EXPECTED_SPEECH_TASK_ID,
        EXPECTED_SINGING_TASK_ID,
    ):
        raise ValueError(
            "SemBridge task-token IDs differ from the released checkpoint: "
            f"speech={speech_id}, singing={singing_id}"
        )
    return tokenizer


def load_text_tokenizer(path_or_id: str | Path, *, revision: str | None = None):
    """Load the checkpoint tokenizer and repair legacy archives missing SVS tokens."""

    try:
        from transformers import LlamaTokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers is required for text tokenization") from exc

    kwargs: dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(str(path_or_id), **kwargs)
    except AttributeError as exc:
        if "keys" not in str(exc):
            raise
        kwargs["extra_special_tokens"] = {}
        tokenizer = LlamaTokenizerFast.from_pretrained(str(path_or_id), **kwargs)
    if len(tokenizer) < EXPECTED_TEXT_VOCAB_SIZE:
        extend_svs_vocabulary(tokenizer)
    else:
        speech_id = int(tokenizer.convert_tokens_to_ids(SPEECH_TASK_TOKEN))
        singing_id = int(tokenizer.convert_tokens_to_ids(SINGING_TASK_TOKEN))
        if (len(tokenizer), speech_id, singing_id) != (
            EXPECTED_TEXT_VOCAB_SIZE,
            EXPECTED_SPEECH_TASK_ID,
            EXPECTED_SINGING_TASK_ID,
        ):
            raise ValueError(
                "tokenizer is incompatible with "
                "SemBridge-Unified-0722-BigVGAN-Large-400k"
            )
    return CharacterAwareTokenizer(tokenizer)


class CharacterAwareTokenizer:
    """Preserve the training-time single-character handling for Chinese text."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.multichar_tokens = {
            token
            for token in tokenizer.get_vocab()
            if len(token) >= 2 and all("\u4e00" <= char <= "\u9fff" for char in token)
        }

    def __len__(self) -> int:
        return len(self.tokenizer)

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        output: list[str] = []
        for token in self.tokenizer.tokenize(text):
            clean = token.replace("▁", "")
            output.extend(list(clean) if clean in self.multichar_tokens else [token])
        return output

    def encode(self, text: str) -> list[int]:
        return [int(value) for value in self.tokenizer.convert_tokens_to_ids(self.tokenize(text))]

    def task_token_id(self, task: str) -> int:
        normalized = task.lower()
        if normalized in {"tts", "speech"}:
            return EXPECTED_SPEECH_TASK_ID
        if normalized in {"svs", "singing"}:
            return EXPECTED_SINGING_TASK_ID
        raise ValueError(f"unknown SemBridge task {task!r}")

    def save_pretrained(self, destination: str | Path) -> None:
        self.tokenizer.save_pretrained(str(destination))


@dataclass(frozen=True)
class Note:
    text: str
    pitch: int
    duration: float
    note_type: int | None = None


def _duration_frames(duration: float, frame_duration_ms: float = 25.0) -> int:
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    frames = max(1, int(round(float(duration) / (frame_duration_ms / 1000.0))))
    if frames >= 500:
        raise ValueError(f"note duration maps to unsupported frame count {frames} (maximum 499)")
    return frames


def _accumulated_duration_frames(
    duration: float, frame_duration_ms: float = 25.0
) -> int:
    """Round cumulative time without applying the per-note vocabulary limit."""

    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    return max(1, int(round(float(duration) / (frame_duration_ms / 1000.0))))


def parse_notes(values: Sequence[dict[str, Any] | Note]) -> list[Note]:
    notes: list[Note] = []
    for value in values:
        if isinstance(value, Note):
            note = value
        elif isinstance(value, dict):
            note = Note(
                text=str(value.get("text", "")),
                pitch=int(value["pitch"]),
                duration=float(value["duration"]),
                note_type=(int(value["note_type"]) if value.get("note_type") is not None else None),
            )
        else:
            raise TypeError("notes must contain objects")
        if not 0 <= note.pitch <= 128:
            raise ValueError(f"note pitch {note.pitch} is outside [0, 128]")
        if note.duration <= 0:
            raise ValueError("note duration must be positive")
        notes.append(note)
    if not notes:
        raise ValueError("singing input requires at least one note")
    return notes


def format_svs_notes(
    values: Sequence[dict[str, Any] | Note], *, frame_duration_ms: float = 25.0
) -> str:
    """Format symbolic notes exactly as the 0722 Lance text processor."""

    notes = parse_notes(values)
    if all(note.note_type is None for note in notes):
        return "".join(
            f"<|note_pitch_{note.pitch}|>"
            f"<|note_dur_{_duration_frames(note.duration, frame_duration_ms)}|>"
            f"{note.text}"
            for note in notes
        )

    units: list[tuple[str, str]] = []
    previous_frames = 0
    duration_accum = 0.0
    for note in notes:
        duration_accum += note.duration
        accumulated = _accumulated_duration_frames(duration_accum, frame_duration_ms)
        frames = max(1, accumulated - previous_frames)
        previous_frames += frames
        if frames >= 500:
            raise ValueError(f"note duration maps to unsupported frame count {frames}")
        pair = f"<|note_pitch_{note.pitch}|><|note_dur_{frames}|>"
        if note.note_type == 3 and units:
            previous_pair, previous_text = units[-1]
            units[-1] = (previous_pair + pair, previous_text)
        else:
            units.append((pair, note.text))
    return "".join(pair + text for pair, text in units)


def encode_segments(
    tokenizer: CharacterAwareTokenizer,
    segments: Iterable[dict[str, Any]],
    *,
    frame_duration_ms: float = 25.0,
) -> tuple[list[int], list[int]]:
    """Encode speech/singing segments and return aligned text and task IDs."""

    text_ids: list[int] = []
    task_ids: list[int] = []
    for segment in segments:
        task = str(segment.get("task", "")).lower()
        if task in {"tts", "speech"}:
            text = str(segment.get("text", ""))
            if not text:
                raise ValueError("speech segment requires non-empty text")
            segment_ids = tokenizer.encode(text)
            task_id = tokenizer.task_token_id("speech")
        elif task in {"svs", "singing"}:
            text = format_svs_notes(
                segment.get("notes", []), frame_duration_ms=frame_duration_ms
            )
            segment_ids = tokenizer.encode(text)
            task_id = tokenizer.task_token_id("singing")
        else:
            raise ValueError(f"unknown segment task {task!r}")
        text_ids.extend(segment_ids)
        task_ids.extend([task_id] * len(segment_ids))
    if not text_ids:
        raise ValueError("at least one non-empty segment is required")
    return text_ids, task_ids
