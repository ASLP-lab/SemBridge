"""Shape, masking, and causal-label contracts for SemBridge sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

IGNORE_INDEX = -100
T = TypeVar("T")


def latent_frame_count(num_samples: int, hop_length: int = 1764) -> int:
    """Return the number of complete causal latent frames for a waveform."""

    if num_samples < 0 or hop_length <= 0:
        raise ValueError("num_samples must be non-negative and hop_length positive")
    return num_samples // hop_length


def valid_patch_mask(frame_mask: Sequence[bool], patch_size: int = 2) -> list[bool]:
    """Return one validity flag per complete patch.

    A patch is valid only when all of its frames are valid. An incomplete tail is
    intentionally dropped, so it can never contribute to semantic alignment.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    complete = len(frame_mask) // patch_size
    return [
        all(frame_mask[i * patch_size : (i + 1) * patch_size])
        for i in range(complete)
    ]


def patch_latents(
    latents: Sequence[Sequence[T]], patch_size: int = 2, latent_dim: int = 64
) -> list[list[T]]:
    """Flatten consecutive 25 Hz latent frames into 12.5 Hz patches.

    Incomplete tails are dropped. Each returned paper-setting patch has 128
    values (``2 x 64``).
    """

    if patch_size <= 0 or latent_dim <= 0:
        raise ValueError("patch_size and latent_dim must be positive")
    for index, frame in enumerate(latents):
        if len(frame) != latent_dim:
            raise ValueError(f"latent frame {index} has {len(frame)} values, expected {latent_dim}")
    complete = len(latents) // patch_size
    return [
        [value for frame in latents[i * patch_size : (i + 1) * patch_size] for value in frame]
        for i in range(complete)
    ]


def build_next_semantic_targets(
    *,
    text_length: int,
    audio_patch_count: int,
    semantic_tokens: Sequence[int],
    prompt_patch_count: int = 0,
    padded_length: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Build causal labels for ``[text, audio_start, patches, audio_end]``.

    ``audio_start`` predicts token 0 and patch ``i`` predicts token ``i+1``.
    Partial prompt/target mixtures were not an archived packing primitive and
    are rejected; a complete prompt record receives only ignore labels.
    """

    if min(text_length, audio_patch_count, prompt_patch_count) < 0:
        raise ValueError("lengths must be non-negative")
    if prompt_patch_count > audio_patch_count:
        raise ValueError("prompt_patch_count cannot exceed audio_patch_count")
    if prompt_patch_count not in {0, audio_patch_count}:
        raise ValueError("partial prompts must be represented as a separate packed record")
    if prompt_patch_count == 0 and len(semantic_tokens) != audio_patch_count:
        raise ValueError("semantic tokens must be aligned one-to-one with audio patches")

    natural_length = text_length + 1 + audio_patch_count + 1
    total_length = natural_length if padded_length is None else padded_length
    if total_length < natural_length:
        raise ValueError("padded_length cannot be shorter than the packed sequence")
    targets = [ignore_index] * total_length
    if prompt_patch_count == audio_patch_count or audio_patch_count == 0:
        return targets

    audio_start = text_length
    targets[audio_start : audio_start + audio_patch_count] = [int(v) for v in semantic_tokens]
    return targets


def build_packed_semantic_targets(
    *,
    task_token_count: int,
    text_length: int,
    audio_patch_count: int,
    semantic_tokens: Sequence[int],
    prompt_patch_count: int = 0,
    padded_length: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Add ignored task-token positions to causal generator labels."""

    if task_token_count < 0:
        raise ValueError("task_token_count must be non-negative")
    natural = task_token_count + text_length + 1 + audio_patch_count + 1
    if padded_length is not None and padded_length < natural:
        raise ValueError("padded_length cannot be shorter than the packed sequence")
    inner_padding = None if padded_length is None else padded_length - task_token_count
    inner = build_next_semantic_targets(
        text_length=text_length,
        audio_patch_count=audio_patch_count,
        semantic_tokens=semantic_tokens,
        prompt_patch_count=prompt_patch_count,
        padded_length=inner_padding,
        ignore_index=ignore_index,
    )
    return [ignore_index] * task_token_count + inner


def resolve_anchor_layer(anchor_layer: int, num_layers: int = 32) -> int:
    """Convert a 1-based paper layer to a 0-based hidden-state index."""

    if anchor_layer < 1 or anchor_layer > num_layers:
        raise ValueError(f"anchor layer must be within [1, {num_layers}]")
    return anchor_layer - 1


def validate_task_sequence(record: dict) -> None:
    """Validate raw or cached TTS, SVS, and mixed conditioning fields."""

    task = record.get("task")
    if task == "tts":
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError("TTS records require non-empty text")
    elif task == "svs":
        if "notes" in record:
            notes = record["notes"]
            if not isinstance(notes, list) or not notes:
                raise ValueError("SVS notes must be a non-empty list")
            for note in notes:
                if not isinstance(note, dict) or any(
                    key not in note for key in ("text", "pitch", "duration")
                ):
                    raise ValueError("each SVS note requires text, pitch, and duration")
                if not 0 <= int(note["pitch"]) <= 128 or float(note["duration"]) <= 0:
                    raise ValueError("SVS pitch/duration is outside the supported range")
            return
        required = ("lyrics", "note_pitches", "note_durations")
        if any(key not in record for key in required):
            raise ValueError("SVS records require notes or aligned lyric/pitch/duration arrays")
        sizes = [len(record[key]) for key in required]
        if len(set(sizes)) != 1 or sizes[0] == 0:
            raise ValueError("SVS lyric, pitch, and duration fields must align")
        if any(duration <= 0 for duration in record["note_durations"]):
            raise ValueError("SVS note durations must be positive")
        onsets = record.get("note_onsets")
        if onsets is not None:
            if len(onsets) != sizes[0]:
                raise ValueError("optional SVS note_onsets must align with notes")
            if any(b < a for a, b in zip(onsets, onsets[1:])):
                raise ValueError("SVS note onsets must be non-decreasing")
        note_types = record.get("note_types")
        if note_types is not None and len(note_types) != sizes[0]:
            raise ValueError("optional SVS note_types must align with notes")
    elif task == "mixed":
        segments = record.get("segments")
        task_ids = record.get("task_token_ids")
        if segments is None and task_ids is not None:
            if not isinstance(task_ids, list) or not task_ids:
                raise ValueError("cached mixed task_token_ids must be non-empty")
            return
        if not isinstance(segments, list) or not segments:
            raise ValueError("mixed records require non-empty segments")
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("mixed segments must be objects")
            segment_task = str(segment.get("task", "")).lower()
            if segment_task in {"tts", "speech"}:
                if not isinstance(segment.get("text"), str) or not segment["text"].strip():
                    raise ValueError("speech segments require non-empty text")
            elif segment_task in {"svs", "singing"}:
                validate_task_sequence({"task": "svs", "notes": segment.get("notes")})
            else:
                raise ValueError(f"unknown mixed segment task {segment_task!r}")
    else:
        raise ValueError("task must be 'tts', 'svs', or 'mixed'")
