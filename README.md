# Spectral-State Navigation

Spectral state navigation and its comparison methods. The model retains the
complete intensity sequence in each spectral patch and encodes it using a
learnable, modality-specific 1D CNN. Methods include Descriptor BO (Mean, PI, EI, LCB,
UCB), Random Forest, TPE-24, Random Search, Descriptor Transformer and TabPFN.

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

Data are read from the project directory by default. Alternatively, use
`--data-dir` to select another directory. Required contents:

```text
Initial_experiments_sol.xlsx
All_experiments_sol.xlsx
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

## Spectral State Navigation

Each measured GPC, NMR and FTIR spectrum is divided into 20 contiguous index
patches. Full intensity sequences are retained, with zero padding only for
short patches; the five handcrafted patch statistics are not used as model
inputs. The CNN has two Conv1d layers: 1 to 16 channels with kernel 7, stride 4
and padding 3, then 16 to 32 channels with kernel 5, stride 4 and padding 2.
Each convolution is followed by GELU. Flattened features are projected to a
96-dimensional token. Each modality has its own CNN; weights are shared across
patches within a modality.

The 60 spectral tokens undergo modality-specific cross-attention, are combined
with an MLP-embedded condition token and are processed by a shared three-layer
Transformer encoder (dimension 96, four attention heads, feed-forward dimension
384, dropout 0.10). The seven outputs are D_GPC, D_NMR, D_FTIR, D_aux, Mw, PDI
and Conversion. D_aux is the sum of the three raw modality distances.

Training uses measured spectra and matched reaction conditions. Fixed target
spectra are constructed from up to 24 initial neighbors in property space and
provide spectral-distance supervision; they are not direct candidate-model
inputs. For an unmeasured candidate, the complete raw-patch vectors of its eight
nearest revealed samples in standardized condition space are combined to
estimate candidate spectral features. These features and the candidate
conditions are then evaluated by the trained model.

The model retains all seven prediction tasks. The mean loss of the three
modality-distance tasks has aggregate weight 0.4; D_aux, Mw, PDI and Conversion
have weights 0.8, 1.8, 2.4 and 2.2. At every round, the model is reinitialized
and trained for 180 epochs using AdamW (learning rate 0.002, weight decay
0.0001, batch size 64, gradient norm clip 1.0), restoring the checkpoint with
minimum validation loss. There is no early stopping. Sample suffixes 01, 05
and 10 of each revealed batch form the validation set.

Ranking uses predicted properties without an explicit spectral-distance term.
Batch score is the minimum of its 12 member scores, and the lowest-scoring
unrevealed batch is selected. Comparison-method implementations are unchanged.
Descriptor BO retains its BayesianRidge/RBF-random-
feature surrogate, not an exact GP; UCB minimizes mu + 0.13 sigma.

These are database-replay programs. Success requires at least one measured
sample with 9000 <= Mw <= 11000, PDI < 1.50 and Conversion > 70%. The replay
protocol and comparison-method implementations are retained. Select a code
license before public distribution.
