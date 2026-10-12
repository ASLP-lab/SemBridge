# Third-Party Notices

SemBridge source code is distributed under [Apache-2.0](LICENSE), except for
upstream portions governed by the notices below. Model weights, tokenizer
artifacts, and runtime dependencies retain their own applicable terms.

## Model Components

- **VoxCPM / OpenBMB**: portions of the causal language model, local encoder,
  and flow-matching speech generator are adapted from
  [VoxCPM](https://github.com/OpenBMB/VoxCPM) and OpenBMB components under
  Apache-2.0. SemBridge adapts task conditioning and semantic anchoring while
  retaining parameter names for checkpoint compatibility.
- **DAC**: the acoustic codec draws on
  [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
  The original MIT copyright and permission notice is retained in
  [LICENSES/DAC-MIT.txt](LICENSES/DAC-MIT.txt).
- **BigVGAN / alias-free-torch**: the implementation in
  `sembridge/models/bigvgan.py` is adapted from
  [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN) and
  [alias-free-torch](https://github.com/junjun3518/alias-free-torch).
  It uses the pure-PyTorch AMPBlock1 path without custom CUDA kernels.
  The BigVGAN MIT notice is in
  [LICENSES/BigVGAN-MIT.txt](LICENSES/BigVGAN-MIT.txt);
  alias-free-torch is Apache-2.0.
  This code repository does not redistribute BigVGAN pretrained weights.

## Research Acknowledgements

We acknowledge [DiTAR](https://arxiv.org/abs/2502.03930) and
[SAC](https://github.com/Soul-AILab/SAC) for their research on speech generation
and semantic-acoustic representations. A research acknowledgement does not
imply that a project's code or weights are included in this package.

## External Semantic Tokenizer

SemBridge uses [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) semantic
supervision during training. Its external semantic tokenizer is **not required
for inference**, and its code and weights are not included in this repository.

The GLM tokenizer has its own model license, including attribution and
redistribution conditions. Consult the
[official model repository](https://huggingface.co/zai-org/glm-4-voice-tokenizer)
for the applicable terms. SemBridge's code license does not replace those terms
or grant additional rights to external assets.

## Runtime Dependencies

PyTorch, torchaudio, Transformers, SoundFile, Pydantic, einops, safetensors,
NumPy, and huggingface-hub retain their own licenses and notices.

## Example Assets

Selected reference recordings and score annotations from the public
[SemBridge Demo](https://tiamojames.github.io/SemBridge_Demo/) are included in
`examples/` for inference. Their source corpora, original sample identifiers,
public audio URLs, and reference hashes are recorded in `examples/index.json`.
These assets are not re-licensed under the source-code Apache-2.0 license;
their original rights and applicable terms remain in effect.
