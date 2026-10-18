# Spectral-State Navigation

Spectral-state navigation is a closed-loop optimization method for identifying
lithographic polymers that satisfy predefined molecular-weight (Mw), dispersity
(PDI) and conversion requirements. Its predictive model, SPA-Former, combines
GPC, NMR and FTIR spectral features with reaction conditions to evaluate
candidate experiments. The model is updated as experimental evidence
accumulates, and the next experimental batch is selected using predicted
properties.

This repository implements spectral-state navigation and its comparison
methods for retrospective database-replay evaluation.

## Method Overview

1. **Define a spectral reference.** Fixed target GPC, NMR and FTIR spectra are
   constructed from the initial revealed experiments using a KNN mapping from
   the specified target property values.
2. **Learn spectral-state representations.** Each spectrum is divided into 20
   patches, represented by five intensity statistics per patch. SPA-Former
   combines these features with reaction conditions through cross-attention
   and a shared three-layer Transformer encoder. It jointly predicts Mw, PDI,
   conversion and four auxiliary spectral-distance outputs relative to the
   fixed target spectra.
3. **Evaluate untested candidates.** Candidate spectral patch features are
   estimated from the eight nearest revealed samples in standardized
   reaction-condition space. The fixed target spectra define the auxiliary
   supervision labels; they are not direct model inputs during candidate
   evaluation. Candidates are ranked by predicted property-window violations
   and deviations from target property values, without an explicit
   spectral-distance term in the ranking score.
4. **Select, reveal and update.** Each batch receives the lowest score among
   its 12 time-resolved candidate states. The lowest-scoring untested batch is
   selected, its stored spectra and measured properties are revealed, and all
   12 states are added to the accumulated training dataset. The procedure
   repeats until a measured sample meets all property criteria or the
   iteration limit is reached.

Spectral information therefore contributes through the model inputs and
auxiliary training objectives, while measured properties determine final
target attainment. Database replay does not perform new physical experiments.

## Comparison Methods

Descriptor BO (Mean, PI, EI, LCB and UCB), Random Forest, TPE-24, Random Search,
Descriptor Transformer and TabPFN are included for comparison.

## Files

- `spa_former/`: spectral state navigation model and training.
- `baselines/`: comparison methods and their required configuration.
- `benchmark/`: shared data handling, generated seeds and comparisons.
- `spectral_state_navigation.py`: main-method entry point.
- `spa_former/spa-former.py`: model program, also directly runnable.
- `run_benchmark.py`: single or multiple comparison methods, including spectral state navigation.

## Install

Use Python 3.9 or newer and an appropriate PyTorch CPU/CUDA build.

```bash
python -m pip install -r requirements.txt
```

TabPFN also requires its external pretrained weights and any associated access
permissions. These weights and credentials are not included.

## Data

Download the GPC, NMR and FTIR spectral dataset from this repository's
[GitHub Releases](https://github.com/PILABAI4S/Spectral-state-navigation/releases).
Extract the spectral data archives into the project directory so that the
`GPC/`, `NMR/` and `FTIR/` folders sit alongside the two experiment workbooks
included in this repository.

Data are read from the project directory by default. Alternatively, use
`--data-dir` to select another directory. Required contents:

```text
Initial_experiments.xlsx
All_experiments.xlsx
GPC/<batch>-01.csv ... <batch>-12.csv
NMR/<batch>-01.csv ... <batch>-12.csv
FTIR/<batch>-01.csv ... <batch>-12.csv
```

Workbook identifiers are `Reaction` (Initial) and `Reaction_ID` (All). Required
columns: `Total_mols`, `Initiator`, `Time`, `Temp`, `Mw`, `PDI`, `Conversion`.
Units are mmol, min, degrees Celsius, Da and percent as appropriate. Each batch
contains 12 samples at 120-min intervals. Initial selects revealed IDs; All
supplies their actual values. Baselines need only the two workbooks.

GPC/FTIR files have two comma-separated columns and one header row. NMR files
have two whitespace-separated columns without a header. Columns are spectral
coordinate and intensity/transmittance. Keep the two workbooks with the three
spectral folders when distributing a combined code-and-data package.

## Run

```bash
# Spectral state navigation only
python spectral_state_navigation.py --data-dir /path/to/dataset --output results/spectral_state_navigation

# Spectral state navigation and selected comparison methods
python run_benchmark.py --methods spectral_state_navigation descriptor_bo_ucb descriptor_transformer --data-dir /path/to/dataset --output results/comparison

# All methods
python run_benchmark.py --methods all --data-dir /path/to/dataset --output results/all
```

With data beside the program, a short local installation check is:

```bash
python spectral_state_navigation.py --repetitions 1 --smoke
```

Smoke mode runs at most two rounds with three epochs per round. It verifies
execution only and is not a complete optimization performance experiment.

Defaults: 20 fresh random seeds and a 200-round cap. The shared runner uses
the same newly generated seeds for every selected method. Historical seeds
are not bundled. Generated seeds and results stay in local output folders,
which are excluded from Git. Use `--help` for available options.

The model retains seven prediction tasks and 180 epochs per round, restoring the
minimum-validation-loss checkpoint. Ranking uses predicted properties without
a spectral-distance term. Descriptor BO retains its BayesianRidge/RBF-random-
feature surrogate, not an exact GP; UCB minimizes mu + 0.13 sigma.

These are database-replay programs. Success requires at least one measured
sample with 9000 < Mw < 11000, PDI < 1.50 and Conversion > 70%. Numerical
algorithms and method-specific preprocessing are unchanged. Select a code
license before public distribution.
