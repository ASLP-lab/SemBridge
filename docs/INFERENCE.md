# Inference

The model is `ASLP-lab/SemBridge`. Inference needs only this checkpoint and
this package, not the external GLM semantic tokenizer used for training.
BigVGAN uses PyTorch operations without custom CUDA kernels.

This package supports the released model-directory format. Training checkpoints
must be exported to that format before loading; training and fine-tuning code
are not included in this release.

## Prepared Examples

Start with the [Demo examples](../examples/README.md). Each request
already contains its reference transcript, aligned singing notes where needed,
and generation settings. The reference WAV files are included in the repository.

```bash
bash scripts/infer_examples.sh --list
bash scripts/infer_examples.sh --task svs
bash scripts/infer_examples.sh --task unified
bash scripts/infer_examples.sh --example svs_zh_r001 unified_zh_novel_wukong
```

The runner loads the model once and writes `outputs/examples/<example_id>.wav`.
No MIDI extraction or manual lyric alignment is needed for these examples.

## Python API

```python
from sembridge import SemBridge
from sembridge.requests import load_request

model = SemBridge.from_pretrained(
    "ASLP-lab/SemBridge", device="cuda:0", dtype="bfloat16"
)
model.generate(
    output="speech.wav", **load_request("examples/tts/zh_r001.json"),
)
model.generate(
    output="singing.wav", decoder="bigvgan", **load_request("examples/svs/zh_r001.json"),
)
```

`device="cuda:0"` selects the first visible GPU; use `cuda:1`, `cuda:2`, etc.
to select another. Indices follow `CUDA_VISIBLE_DEVICES` when it is set.
`generate()` returns `(waveform, generated_patches)` as float32 CPU tensors.
Set `output` to also save a PCM16 WAV, or omit it to use the tensors directly.

`decoder="auto"` is the default: TTS uses the native causal decoder; SVS and
unified use BigVGAN-Large. Either can be selected explicitly for any task.
Missing explicit decoders raise an error. A legacy single-codec directory
remains usable in `auto` mode with its only codec. Both bundled codecs are
loaded, and encoder weights are verified and shared; explicit selection does
not load only one codec.

Both routes consume 64-D, 25 Hz latents. A patch contains two frames (80 ms).
`max_new_patches=500` / `--max-patches 500` is a 40-second upper bound, not a
requested duration. Generation can stop earlier. A request may include a
`generation` object with `max_new_patches`, `min_new_patches`, `flow_steps`,
`cfg_value`, `temperature`, and `seed`. `load_request` flattens these fields into
Python generation arguments. CLI flags override request settings; unspecified
flow steps, guidance and temperature fall back to `generation_config.json`.
Use at least two flow steps.
`python -m sembridge infer` accepts the same arguments as `sembridge-infer`.

| CLI option | Python argument | Default without a request override |
| --- | --- | --- |
| `--decoder` | `decoder` | `auto` |
| `--max-patches` | `max_new_patches` | `500` |
| `--min-patches` | `min_new_patches` | `2` |
| `--flow-steps` | `flow_steps` | Model config; otherwise `10` |
| `--cfg-value` | `cfg_value` | Model config; otherwise `2.0` |
| `--temperature` | `temperature` | Model config; otherwise `1.0` |
| `--seed` | `seed` | `42` |

## Reference voice

Provide a recording with consent and its matching transcript:

```python
model.generate(
    output="speech.wav", task="tts", text="This is the new sentence.",
    prompt_audio="reference.wav", prompt_text="This is the reference transcript.",
    prompt_task="speech", decoder="bigvgan", seed=42,
)
```

`prompt_task` describes the reference, not the target. Use `speech` for a spoken
reference even when generating SVS, and `singing` for a sung reference. Its
default is `singing` for SVS and `speech` otherwise. The same fields work in
request JSON. Relative `prompt_audio` paths resolve relative to the request
file. Audio is downmixed and resampled to 44.1 kHz. The prepared sung references
already include score-formatted `prompt_text`; keep it paired with the matching
WAV rather than substituting a different lyric transcript.

For example, put this JSON next to your own `reference.wav` and pass it with
`--task tts --input /path/to/request.json`:

```json
{
  "task": "tts",
  "text": "This is the new sentence.",
  "prompt_audio": "reference.wav",
  "prompt_text": "This is the reference transcript.",
  "prompt_task": "speech"
}
```

The seed covers both sampled prompt latents and generation. Paired decoder
comparisons need the same checkpoint, request, seed and generation settings
in the same environment. `model.generate(...)` returns waveform and generated
patches. Bitwise reproducibility across devices/versions is not guaranteed.

## SVS and Mixed Inference

Each note contains `text`, MIDI `pitch`, and `duration` **in seconds**.
Durations are internally quantized in 25 ms units; do not pass frame counts
as seconds. See [the prepared SVS request](../examples/svs/zh_r001.json).
The supplied `note_type` labels preserve the original alignment: `3` extends
the preceding lyric over another note. `<SP>` marks a rest. Do not remove these
fields or merge repeated lyrics indiscriminately.

These inputs are aligned score annotations, not raw MIDI files. A plain MIDI
file may not contain lyric-to-note alignment or identify the intended vocal
track. This package does not automatically transcribe audio, extract MIDI, or
align arbitrary lyrics. To use a new song, prepare those annotations first;
the bundled examples require none of that work.

Mixed generation accepts ordered speech/singing `segments`; see
[the mixed request](../examples/unified/zh_novel_wukong.json). Invoke with `--task unified`.
Both `"task": "mixed"` and `"task": "unified"` work at the top level of JSON.

```python
model.generate(
    output="mixed.wav", **load_request("examples/unified/zh_novel_wukong.json"),
)
```

```bash
sembridge-infer --model ASLP-lab/SemBridge --task unified \
  --input examples/unified/zh_novel_wukong.json --output outputs/unified.wav \
  --device cuda:0 --dtype bfloat16 --decoder auto
```

## Local, pinned and offline models

`--model /path/to/model` loads a local directory. For a Hub model, use
`--revision COMMIT_ID`, `--cache-dir PATH`, and `--local-files-only` as needed.
The API accepts `revision`, `cache_dir` and `local_files_only` too.

A complete dual-decoder model directory contains:

```text
config.json
model.safetensors
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
savae_config.json
audiovae.safetensors
unified_savae_config.json
unified_audiovae.safetensors
```

Keep the paired configuration and weight files together. A single
`model.safetensors` file is not a complete inference model.

For generation, use CUDA with BF16. CPU requires `--dtype float32` and is much
slower. The example script loads the model once and executes requests sequentially on one GPU.
On CUDA hardware without BF16 support, use `--dtype float32` (with higher memory
use). The script accepts `MODEL`, `OUTPUT_DIR`, `DECODER`, `DEVICE`, `DTYPE`,
`CFG_VALUE`, `FLOW_STEPS`, `TEMPERATURE`, `SEED`, and `PYTHON_BIN` environment
variables. For CPU use, set both `DEVICE=cpu` and `DTYPE=float32`.
