"""Local and Hugging Face model-directory resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_pretrained_path(
    path_or_id: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    local = Path(path_or_id).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"{path_or_id!r} is not a local path; install huggingface-hub to download it"
        ) from exc
    downloaded = snapshot_download(
        repo_id=str(path_or_id),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    return Path(downloaded).resolve()
