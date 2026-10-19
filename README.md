# Spectral-State Navigation

Spectral-state navigation is a closed-loop optimization method for identifying
lithographic polymers with desired molecular weight (Mw), dispersity (PDI) and
conversion. Its model, SPA-Former, combines GPC, NMR and FTIR spectra with
reaction conditions using learnable 1D CNN spectral encoders and a
cross-attention Transformer.

This repository provides the database-replay implementation and comparison
methods. Detailed methodology is described in the paper and Supplementary
Information.

## Installation

Use Python 3.9 or newer with a suitable PyTorch CPU/CUDA build. A CUDA-capable
GPU is recommended.

```bash
python -m pip install -r requirements.txt
```

TabPFN additionally requires access to its pretrained weights.

## Data

The experiment workbooks are included in this repository. Download the GPC,
NMR and FTIR spectral archives from
[GitHub Releases](https://github.com/PILABAI4S/Spectral-state-navigation/releases)
and extract them into the project directory:

```text
Initial_experiments.xlsx
All_experiments.xlsx
GPC/
NMR/
FTIR/
```

## Usage

Run the following commands from the project directory.

```bash
# Spectral-state navigation
python spectral_state_navigation.py --initial-file Initial_experiments.xlsx --all-file All_experiments.xlsx

# Spectral-state navigation and all comparison methods
python run_benchmark.py --methods all --initial-file Initial_experiments.xlsx --all-file All_experiments.xlsx
```

By default, runs use 20 freshly generated random seeds and a maximum of 200
closed-loop iterations. Comparison methods share the same generated seeds.
Seeds and results are saved locally under `results/`. Add `--repetitions 1
--smoke` for a short execution check, or use `--help` for available options.

These programs replay existing experimental measurements; they do not control
laboratory instruments or execute new physical experiments.
