<div align="center">
  <h1>
    SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation
  </h1>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg?logo=python&logoColor=white" alt="Python">
    <a href="https://arxiv.org/submit/7924473/view"><img src="https://img.shields.io/badge/arXiv-Preview-b31b1b.svg?logo=arXiv" alt="arXiv preview"></a>
    <a href="https://tiamojames.github.io/SemBridge_Demo/"><img src="https://img.shields.io/badge/Demo-Page-orange.svg" alt="Demo"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
  </p>

  <p align="center">
    <i>Training-only semantic-token anchoring for continuous-latent autoregressive speech generation.</i>
  </p>
</div>

## 📖 Introduction

SemBridge is a continuous-latent autoregressive speech generation framework that strengthens linguistic modeling through explicit semantic supervision while keeping inference lightweight.

Continuous acoustic latents preserve rich prosody, timbre, and fine-grained speech detail, but purely continuous autoregressive modeling can make semantic control harder. SemBridge addresses this by introducing a semantic-token anchoring objective during training: continuous acoustic representations and causal language-model hidden states are aligned through a shared semantic-token interface, encouraging the generator to maintain a clearer connection between text semantics and acoustic generation.

At inference time, the semantic anchoring branch is removed. SemBridge therefore generates speech autoregressively over continuous acoustic latents only, without requiring semantic-token generation or changing the runtime decoding procedure.

<div align="center">
  <img src="asset/sembridge.png" alt="SemBridge model overview" width="95%">
</div>

## 🚀 News

- **[2026-08]**: Released the [arXiv paper](https://arxiv.org/submit/7924473/view).

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

Model weights, tokenizer assets, datasets, and third-party components will follow their respective license terms when released.
