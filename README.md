# Random Fuzzy Rules

Random Fuzzy Rules (RFR) is a scikit-learn-compatible classifier that learns compact fuzzy rule sets through randomized candidate generation and efficient Numba-based evaluation. This repository contains the software package and the experiments used to evaluate its ablation settings, selected default configuration, scalability, and predictive performance.

## Reproducing the experiments

Two equivalent wrapper scripts are provided:

- `scripts/reproduce.ps1` for Windows 11 and PowerShell;
- `scripts/reproduce.sh` for Linux and Bash.

The scripts:

1. locate the repository root;
2. check whether `uv` is available;
3. install the recommended `uv 0.11.29` with the official installer if `uv` is missing;
4. synchronize the Python 3.12 and Python 3.10 projects from their lockfiles;
5. run import smoke tests;
6. execute the selected experiment;
7. store replicated outputs separately from the reference results;
8. record an environment manifest, installed package lists, an execution log, and a run summary.

The scripts do not modify or remove the reference outputs under `results/`. Replicated outputs are written to `results_replication/`, which should be excluded from version control.

## Requirements

- Internet access for the first environment setup and for downloading UCI datasets.
- Windows 11 with Windows PowerShell or PowerShell 7, or a Linux distribution with Bash.
- On Linux, one of `curl` or `wget` is required only when `uv` is not already installed.
- Administrator or root privileges are not required by the scripts. If a Linux downloader is missing, install one manually. For example, on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install curl ca-certificates
```

The scripts use the existing `uv` installation when available. `uv 0.11.29` is recommended because it was used for the reference experiments. If another version is detected, the scripts display a warning and continue. If synchronization fails, install `uv 0.11.29` and rerun the command.

## Experiments

The first positional argument selects one of the following operations:

- `ablation`: all one-factor-at-a-time RFR ablation studies;
- `validation`: validation of the selected RFR default configuration;
- `scalability`: sample-count, feature-count, and candidate-count scalability studies;
- `comparison`: final comparison of the six interpretable classifiers;
- `all`: run ablation, validation, scalability, and comparison in that order.

The `validation` experiment is the shortest complete experimental workflow and is recommended as the first reproduction test.

## Options

On Linux, use the GNU-style forms shown below. On Windows PowerShell, use the native switch forms `-Clean`, `-Resume`, `-PlotsOnly`, and `-SkipWarmup`.

- `--clean`: remove replicated outputs for the selected experiment before running it again;
- `--resume`: retain replicated CSV files and skip completed units of work;
- `--plots-only`: regenerate plots from replicated CSV files without fitting estimators;
- `--skip-warmup`: pass the corresponding option to the Python experiment modules.

`--resume` is the default when neither `--clean` nor `--resume` is specified. The scripts print a notice explaining this behavior.

The following combinations are rejected:

- `--clean` together with `--resume`;
- `--clean` together with `--plots-only`.

## Windows 11

From the repository root, run a short end-to-end reproduction test with:

```powershell
powershell -ExecutionPolicy Bypass -File `
    .\scripts\reproduce.ps1 `
    validation `
    -Clean
```

If PowerShell 7 is installed, the same script can be run with:

```powershell
pwsh -File .\scripts\reproduce.ps1 validation -Clean
```

The execution-policy override applies only to this process. A global change to the PowerShell execution policy is not required or recommended.

Run every experiment from scratch with:

```powershell
powershell -ExecutionPolicy Bypass -File `
    .\scripts\reproduce.ps1 `
    all `
    -Clean
```

Resume an interrupted comparison with:

```powershell
powershell -ExecutionPolicy Bypass -File `
    .\scripts\reproduce.ps1 `
    comparison `
    -Resume
```

Regenerate comparison plots only with:

```powershell
powershell -ExecutionPolicy Bypass -File `
    .\scripts\reproduce.ps1 `
    comparison `
    -PlotsOnly
```

## Linux

From the repository root, run a short end-to-end reproduction test with:

```bash
bash scripts/reproduce.sh validation --clean
```

Run every experiment from scratch with:

```bash
bash scripts/reproduce.sh all --clean
```

Resume an interrupted comparison with:

```bash
bash scripts/reproduce.sh comparison --resume
```

Regenerate comparison plots only with:

```bash
bash scripts/reproduce.sh comparison --plots-only
```

The script can optionally be marked as executable:

```bash
chmod +x scripts/reproduce.sh
./scripts/reproduce.sh validation --clean
```

## Output directories

Reference results distributed with the repository remain under:

```text
results/
```

Newly reproduced outputs are stored under:

```text
results_replication/
├── ablation/
├── scalability/
├── comparison/
└── reproduction/
```

Every wrapper invocation creates a timestamped directory under `results_replication/reproduction/` containing:

- `environment.json`: operating system, CPU count, memory, Python, `uv`, thread settings, and execution options;
- `main_packages.txt`: installed packages in the Python 3.12 project;
- `gpr_packages.txt`: installed packages in the Python 3.10 project;
- `execution.log`: combined wrapper and experiment output;
- `summary.json`: final status, duration, selected experiment, and completed stages.


## Manual execution

Every Python experiment module also accepts `--results-root`. This makes it possible to reproduce an experiment without the wrapper scripts while still keeping outputs separate from the reference results.

For example:

```powershell
uv run `
    --project environments/main `
    --locked `
    python -m experiments.comparison.run `
    --results-root results_replication
```

```bash
uv run \
    --project environments/main \
    --locked \
    python -m experiments.comparison.run \
    --results-root results_replication
```

The default output root for manual module execution is `results/`.

## Reproducibility expectations

The locked environments, fixed random seeds, common cross-validation folds, and single-thread settings are intended to make model-quality results reproducible. Training times are hardware-dependent and are not expected to match the reference machine. Small floating-point differences may occur across processors and operating systems, but the main Accuracy rankings and statistical conclusions should remain stable.

## Troubleshooting

### `uv` is not found after installation

Open a new terminal and rerun the script. Typical user-level installation locations include `$HOME/.local/bin` on Linux and `%USERPROFILE%\.local\bin` on Windows.

### Environment synchronization fails

Check the reported `uv` version. The recommended version is `0.11.29`. The scripts use `uv sync --locked` and never regenerate the lockfiles.

### Plot-only mode cannot find data

`--plots-only` reads only from `results_replication/`. Run the corresponding experiment first, or copy a compatible replicated CSV into the expected location. The scripts intentionally do not read from or modify the reference `results/` directory.

### An experiment was interrupted

Run the same experiment again with `--resume`, or omit both mode flags because resume is the default. Existing successful dataset, fold, and estimator combinations are skipped where supported by the Python modules.

## Third-party and derived software

The repository includes `packages/gpr-fast`, a modified and optimized
implementation derived from the original GPR project: [gpr-algorithm](https://github.com/czmilanna/gpr-algorithm).

The original project is licensed under the MIT License. Its copyright
notice and license terms are preserved in:

`packages/gpr-fast/LICENSE`

See `packages/gpr-fast/README.md` for the original repository reference
and a summary of the modifications.

## Use of artificial intelligence

GPT-5.6 was used as an AI-assisted development tool during the preparation of
this repository. Its assistance included the generation of initial code drafts,
code review, refactoring suggestions, documentation drafting, experimental
workflow design, and troubleshooting of the reproducibility scripts.

All AI-generated code and suggestions were reviewed, analyzed, adapted where
necessary, and tested by the authors. The authors remain responsible for the
final source code, experimental methodology, reported results, and scientific
conclusions.