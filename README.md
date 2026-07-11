# When GNNs Beat MLPs

[![Version](https://img.shields.io/badge/version-0.1.0-yellow.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

A reproducible benchmark measuring the predictive contribution of
graph structure. Three graph neural networks are compared against multilayer
perceptron baselines that receive the same input features but no graph
structure.

| Task | Dataset | GNN | Baseline |
|---|---|---|---|
| Node classification | Cora | GCN | MLP |
| Link prediction | Amazon Computers | GAE | MLP encoder |
| Graph classification | NCI1 | GIN | MLP + shared readout |

Each comparison keeps optimization, input features, and task head identical across
the two models, and the baseline is deliberately given the larger parameter budget.
Every experiment runs over a fixed set of ten seeds, where each seed governs every
stochastic component of its run (initialization, sampled splits, negative
sampling), so a run is determined by its seed. Results are reported as
mean and standard deviation, alongside parameter counts.

The difference in performance estimates the contribution of relational inductive bias.

## Setup & Installation

### Prerequisites

- Python 3.12+.
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- No GPU is required. PyTorch is pinned to the CPU-only build on every platform by design.
The benchmarked models are small enough to train on CPU, while a CPU-only build, together
with fixed seeds and configuration, supports the reproducibility of the benchmark.
Because the same build is used across operating systems, `uv.lock` fully describes the
compute environment rather than deferring part of it to each machine.

### Local Installation

The committed `uv.lock` file pins the exact package versions required to reproduce the
benchmark environment.

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/roymacias/when-gnns-beat-mlps.git
cd when-gnns-beat-mlps

# 3. Create the virtual environment and install dependencies
make setup
```

## Usage

```bash
make data        # download benchmarks and materialize seeded splits
make node        # GCN vs. MLP on Cora        (train + evaluate)
make link        # GAE vs. MLP on Amazon      (train + evaluate)
make graph       # GIN vs. MLP on NCI1        (train + evaluate)
make all         # end-to-end pipeline
```

`make help` lists every target. Development tasks: `make format`, `make lint` (ruff and format
check), `make typecheck` (mypy) and `make test`.

## License

Released under the [MIT License](LICENSE).
