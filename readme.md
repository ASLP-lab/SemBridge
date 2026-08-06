
<div align="center">
  <h1>
    SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation
  </h1>

  <p align="center">
    <a href="#-installation"><img src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    <a href="https://sembridge.github.io/"><img src="https://img.shields.io/badge/🌐%20Demo-Page-orange.svg" alt="Demo"></a>
    <a href="https://huggingface.co/YOUR_ORG/SemBridge"><img src="https://img.shields.io/badge/🤗%20HF-Model-yellow.svg" alt="HF Model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
  </p>

  <p align="center">
    <i>Training-only semantic-token anchoring for continuous-latent autoregressive speech generation.</i>
  </p>
</div>

## 📖 Introduction

SemBridge is a continuous-latent autoregressive speech generation framework that introduces explicit semantic supervision without changing the inference procedure.

It uses a shared semantic-token interface to align continuous acoustic representations and anchor the hidden states of the causal language model during training. The semantic anchoring branch is removed at inference, preserving autoregressive generation over continuous acoustic latents only.


For more details, please refer to our paper:

**SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation**

<div align="center">
  <img src="docs/static/images/sembridge.png" alt="SemBridge architecture" width="90%">
</div>

## 🚀 News

- **[2026-08]**: Released the initial SemBridge checkpoint and zero-shot TTS codebase.

## 🗺️ Roadmap

- [x] Release the official SemBridge checkpoint
- [x] Release zero-shot TTS inference code
- [ ] Release TTS fine-tuning and data-preparation code
- [ ] Release unified TTS and SVS inference
- [ ] Release SVS fine-tuning code
- [ ] Release additional checkpoints, examples, and evaluation tools

## ⚙️ Installation

We recommend using Conda to manage the environment.

```bash
git clone https://github.com/YOUR_ORG/SemBridge.git
cd SemBridge

conda create -n sembridge python=3.10
conda activate sembridge

pip install -e .
```

SemBridge requires Python 3.10+ and PyTorch 2.6+.

## 📦 Model Checkpoint

The official SemBridge checkpoint is available on Hugging Face:

[SemBridge 🤗](https://huggingface.co/YOUR_ORG/SemBridge)

The checkpoint can be downloaded automatically through `SemBridge.from_pretrained(...)` or loaded from a local model directory.

## 🚀 TTS Inference

SemBridge performs zero-shot TTS using a reference audio prompt and its transcription.

```python
from sembridge import SemBridge

model = SemBridge.from_pretrained(
    "YOUR_ORG/SemBridge",
    device="cuda",
    dtype="bfloat16",
)

model.generate(
    output_path="speech.wav",
    text="Semantic anchoring improves linguistic fidelity.",
    prompt_audio="prompt.wav",
    prompt_text="This is the transcription of the reference audio.",
    cfg_value=2.0,
)
```

Example inference requests and scripts are provided in the repository.

```bash
bash scripts/infer_examples.sh
```

Generated audio will be saved under:

```text
artifacts/infer_examples/
```

## 🙏 Acknowledgements

SemBridge uses components and resources from several open-source projects, including [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice), [PyTorch](https://pytorch.org/), and [Hugging Face](https://huggingface.co/).

We sincerely thank the authors and contributors for their valuable open-source work.

## 📝 Citation

If you find SemBridge useful in your research, please consider citing our paper:

```bibtex
@misc{xie2026sembridge,
  title  = {SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation},
  author = {Hanke Xie and Haopeng Lin and Jiale Qian and Dake Guo and Yuepeng Jiang and Zhichao Wang and Wenxiao Cao and Jingbin Hu and Guobin Ma and Wenhao Li and Huakang Chen and Chengyou Wang and Ming Tao and Zhonghua Fu and Lei Xie and Xinsheng Wang},
  year   = {2026}
}
```

## ⚠️ Responsible Use

Please obtain consent before using a reference voice and clearly disclose synthetic media where appropriate. Do not use this project for impersonation, fraud, harassment, or deceptive content.

## 📜 License

The code in this repository is released under the [Apache-2.0 License](LICENSE).

Model weights, tokenizer assets, datasets, and third-party dependencies are subject to their respective license terms.
