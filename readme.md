<div align="center">
  <h1>
    SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation
  </h1>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg?logo=python&logoColor=white" alt="Python">
    <a href="https://arxiv.org/pdf/2608.07462"><img src="https://img.shields.io/badge/arXiv-2608.07462-b31b1b.svg?logo=arXiv" alt="arXiv paper"></a>
    <a href="https://tiamojames.github.io/SemBridge_Demo/"><img src="https://img.shields.io/badge/Demo-Page-orange.svg" alt="Demo"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
  </p>

  <!-- <p align="center">
    <i>Semantic-token anchoring for continuous-latent autoregressive speech generation.</i>
  </p> -->
</div>

## 📖 Introduction

SemBridge is a semantic-token anchoring framework for continuous-latent autoregressive speech generation. It uses a shared semantic tokenizer to supervise autoregressive LM states with discrete semantic tokens and align continuous acoustic latents, improving linguistic fidelity while maintaining high-quality continuous speech generation. For more details, please refer to our paper.
<div align="center">
  <img src="asset/sembridge.png" alt="SemBridge model overview" width="95%">
</div>

## 🚀 News

- **[2026-08]**: Released the [arXiv paper](https://arxiv.org/pdf/2608.07462).

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
