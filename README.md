# GOLLuM: Gaussian Process Optimized LLMs – Reframing LLMs as Principled Bayesian Optimizers 🧙‍♂️📈

**GOLLuM – Gaussian Process Optimized LLMs are here!**  
*One representation to rule them all!*

> 📄 Paper: [![arXiv](https://img.shields.io/badge/arXiv-2504.06265-b31b1b.svg)](https://arxiv.org/abs/2504.06265)

---

## 🔍 Overview

🎯 GOLLuM addresses the challenge of harnessing LLMs for optimization under uncertainty by introducing:

- LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both
- LLMs to provide a rich and flexible input space for Bayesian optimization
- GPs to model this space with predictive uncertainty for more efficient sampling

🌌 The framework enables a bidirectional feedback loop:
1. The GP guides updates to LLM weights to produce more effective embeddings
2. These embeddings enhance the GP's probabilistic modeling

---

## 🧠 Key Features

- **Unified Representation Learning**: Uses textual templates to represent heterogeneous parameter types (categorical, numerical, structural)
- **GP-Guided LLM Finetuning**: Optimizes LLM embeddings through GP marginal likelihood
- **Implicit Contrastive Learning**: Automatically organizes the latent space into distinct performance regions
- **Chemical reasoning in the latent space**: Uncovering chemical patterns under extreme low-data regimes
- **Architecture Agnostic**: Works with various LLM architectures (encoder, decoder, encoder-decoder)
- **Domain Agnostic**: No requirement for domain-specialized models or pretraining


---

## 🚀 Quickstart

### 📦 Project Dependencies

You can install the environment from a file:

```bash
# Recommended (Conda)
conda env create -f environment.yaml
conda activate gollum

# OR (pip-only)
pip install -r requirements.txt
```

For manual setup or more details, see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

---

### 🛠 Install GOLLuM in editable mode

```bash
pip install -e .
```

---

### ⚙️ Running Experiments

All configuration files for reproducing experiments are included in the `configs/` directory. You can launch an experiment with:

```bash
python train.py --config=configs/pllm_phi.yaml
```

Replace `pllm_phi.yaml` with other config files for variants such as `llm_phi.yaml`, `pllm.yaml`, etc.

---

## 📚 Citation

```bibtex
@inproceedings{
rankovic2025gollum,
title={{GOLL}uM: Gaussian Process Optimized {LLM}s {\textemdash} Reframing {LLM} Finetuning through Bayesian Optimization},
author={Bojana Rankovi{\'c} and Philippe Schwaller},
booktitle={ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling},
year={2025},
url={https://openreview.net/forum?id=2ORViHAUbf}
}
```

---

## ⚖️ License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.

---

## 🤝 Acknowledgements

This work was supported by NCCR Catalysis (grant number 225147), a National Centre of Competence in Research funded by the Swiss National Science Foundation.
