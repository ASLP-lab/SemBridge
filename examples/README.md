# Examples

These 10 cases use the texts, aligned scores, and reference recordings from the
[SemBridge Demo](https://tiamojames.github.io/SemBridge_Demo/). Everything needed
for the input is included: no separate audio download, MIDI extraction, or
manual lyric-to-note alignment is required.

## Run

From the repository root, after installing SemBridge:

```bash
# Inspect the catalog without loading weights
bash scripts/infer_examples.sh --list

# Run one case, one task, or the entire catalog
bash scripts/infer_examples.sh --example svs_zh_r001
bash scripts/infer_examples.sh --task unified
bash scripts/infer_examples.sh

# Select multiple cases and a local checkpoint
MODEL=/path/to/SemBridge bash scripts/infer_examples.sh \
  --example svs_en_r001 unified_zh_novel_wukong
```

The runner loads the model **once**. It defaults to CUDA/BF16 and saves
`outputs/examples/<example_id>.wav`. Use `--output-dir` to choose another
destination. Each JSON also works independently with `sembridge-infer --input`.

## Catalog

| Task | Example / request | Language | Reference audio |
| --- | --- | --- | --- |
| TTS | [tts_zh_r001](tts/zh_r001.json) | Chinese | [WAV](audio/tts_zh_r001.wav) |
| TTS | [tts_zh_r002](tts/zh_r002.json) | Chinese | [WAV](audio/tts_zh_r002.wav) |
| TTS | [tts_en_r001](tts/en_r001.json) | English | [WAV](audio/tts_en_r001.wav) |
| TTS | [tts_en_r002](tts/en_r002.json) | English | [WAV](audio/tts_en_r002.wav) |
| SVS | [svs_zh_r001](svs/zh_r001.json) | Chinese | [WAV](audio/svs_zh_r001.wav) |
| SVS | [svs_zh_r002](svs/zh_r002.json) | Chinese | [WAV](audio/svs_zh_r002.wav) |
| SVS | [svs_en_r001](svs/en_r001.json) | English | [WAV](audio/svs_en_r001.wav) |
| SVS | [svs_en_r002](svs/en_r002.json) | English | [WAV](audio/svs_en_r002.wav) |
| Unified | [unified_zh_robust_caihong](unified/zh_robust_caihong.json) | Chinese | [WAV](audio/unified_zh_robust_caihong.wav) |
| Unified | [unified_zh_novel_wukong](unified/zh_novel_wukong.json) | Chinese | [WAV](audio/unified_zh_novel_wukong.wav) |

Both Unified examples contain **speech, then singing, then speech**, generated
in one sequence. `unified_zh_robust_caihong` uses a sung reference;
`unified_zh_novel_wukong` uses a spoken reference. `prompt_task` is already set
correctly in each request.

## What's Included

- **TTS:** target text, reference recording, and its transcript.
- **SVS:** aligned lyric units, MIDI pitches, durations in seconds, rest and
  note-continuation labels, plus the matching sung reference and its score text.
- **Unified:** ordered speech/singing segments with the same complete score
  representation and reference conditioning.
- **Generation settings:** each request includes its seed, flow steps, guidance,
  temperature, and generation-length bounds. Explicit command-line options take
  precedence; otherwise the single-request CLI and batch runner use the same values.

SVS/Unified examples use 20 flow steps, CFG 2.0, and seed 42. TTS uses 10 flow
steps, CFG 2.0, and seed 42. Output audio is mono 44.1 kHz PCM16 WAV.
These are prepared inputs, not a promise of byte-identical audio to the Demo:
decoder revisions and runtime environments can change the output.

## Your Own Content

For **TTS**, change `text` to your new sentence. Keep `prompt_audio` and
`prompt_text` paired, or replace both with your own reference recording and
transcript.

For **Unified**, the speech segments can be edited directly while keeping a
prepared singing segment. Longer speech may require increasing
`generation.max_new_patches`.

For **a new song**, lyrics must be aligned to the intended melody. A plain MIDI
file may contain accompaniment, multiple tracks, or no aligned lyrics; this
release does not automatically choose a vocal track or align arbitrary words.
Do not simply replace the lyric text while retaining another song's timings.

The score fields are:

| Field | Meaning |
| --- | --- |
| `text` | The lyric unit for this note; `<SP>` marks a rest |
| `pitch` | Integer MIDI pitch; preserve the supplied rest pitch values |
| `duration` | Duration **in seconds**, not MIDI ticks or frame counts |
| `note_type` | Preserve the alignment labels; `3` continues the preceding lyric on another note |

`prompt_text` for a sung reference is already formatted with score tokens.
It should not be edited independently of the reference recording.
See the [inference guide](../docs/INFERENCE.md) for the Python API and full options.

## Sources

The reference audio is copied unchanged from the public Demo. [index.json](index.json)
records the source revision, public audio URLs, and original
sample identifiers. TTS annotations come from Seed-TTS; the singing annotations
come from GTSinger and Opencpop. Spoken character references are identified by
their Demo sample IDs.

These sample assets retain their original rights and terms; they are not covered
by the source-code Apache-2.0 license. See the
[license and third-party notices](../LICENSE).
