# GOLLuM Dependencies

This file documents how to set up the GOLLuM environment for development or reproduction.

## 📦 Requirements (pip)
For pip users, you can install all required packages with:

```bash
pip install -r requirements.txt
```

## 🧪 Reproducible Conda Environment

For a fully reproducible environment using conda:

```bash
conda env create -f environment.yaml
conda activate gollum
```

## Notes

- `torch` is installed via pip for CUDA compatibility
- `rdkit` works from pip in this setup
- `rxnfp` is installed with `--no-deps` to avoid conflicts