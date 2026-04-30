# ThermoMPNN
ThermoMPNN is a graph neural network (GNN) trained using transfer learning to predict changes in stability for protein point mutants.

![ThermoMPNN Scheme](./images/SVG/thermoMPNN_scheme.svg)

For details on ThermoMPNN training and methodology, please see the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.07.27.550881v1). 

## Colab Implementation
For a user-friendly version of ThermoMPNN requiring no installation, use this [Colab notebook](https://colab.research.google.com/drive/1OcT4eYwzxUFNlHNPk9_5uvxGNMVg3CFA#scrollTo=i06A5VI142NT).

## Installation
To install ThermoMPNN, first clone this repository
```
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git
```
Then use the file ```environment.yaml``` install the necessary python dependencies (I recommend using mamba for convenience):
```
mamba env create -f environment.yaml
```
This will create a conda environment called ```thermoMPNN```.

## GEEF Adapter

The repository also includes a lightweight adapter under `geef_adapter/` for scoring multi-mutation enzyme variants inside the GEEF workflow.

### What it does

- accepts a wild-type sequence, variant sequence, structure path, and chain id
- infers the single amino-acid substitutions in each multi-mutant variant
- scores each single substitution with ThermoMPNN
- aggregates the per-mutation ddG values into one variant-level ddG score

Current aggregation behavior:

- `variant_ddg = sum(single_mutation_ddg_i)`

### Minimal example

```python
from pathlib import Path

from geef_adapter import ThermoMPNNConfig, ThermoMPNNScorer, VariantInput

repo_root = Path("/absolute/path/to/ThermoMPNN-1.0.0")

config = ThermoMPNNConfig(
    model_path="/absolute/path/to/thermoMPNN_default.pt",
    local_yaml_path=str(repo_root / "local.yaml"),
    device="cuda",
    default_chain_id="A",
    aggregation_method="sum",
    cache_structures=True,
    combine_variant_mutations=True,
)

scorer = ThermoMPNNScorer(config)

result = scorer.score_variant(
    VariantInput(
        variant_id="demo_variant_001",
        wildtype_sequence="ACDEFGHIKLMNPQRSTVWY",
        variant_sequence="ACDEYGHIKLMNPQKSTVWY",
        pdb_path="/absolute/path/to/structure.pdb",
        chain_id="A",
        mutation_list=None,
        metadata={"source": "demo"},
    )
)

print(result.ddg_sum)
print(result.mutation_list)
print([(item.mutation, item.ddg) for item in result.per_mutation_scores])
```

### Input/output behavior

Input per variant:

- `variant_id`
- `wildtype_sequence`
- `variant_sequence`
- `pdb_path`
- `chain_id`
- optional `mutation_list`
- optional `metadata`

Returned result:

- `ddg_sum`: aggregated variant ddG
- `mutation_list`: normalized single substitutions such as `A123V`
- `per_mutation_scores`: ThermoMPNN score for each single mutation
- `status`, `warnings`, and structured error fields

### Notes

- only substitution mutations are supported right now
- indels are not supported
- structure parsing is cached across variants when possible
- repeated single mutations can be merged across a batch when `combine_variant_mutations=True`

### Local config for GEEF service use

The repo keeps `local.yaml` ignored because it usually contains machine-specific paths.
For a portable starting point, use `local.geef.example.yaml` and copy it to your own `local.yaml` before starting the ThermoMPNN service.

Example:

```bash
cp local.geef.example.yaml local.yaml
# then edit local.yaml to match your machine
```

## Training
The main training script is ```train_thermompnn.py```. To set up a training run, you must write a ```config.yaml``` file (example provided) to specify model hyperparameters. You also must provide a ```local.yaml``` file to tell ThermoMPNN where to find your data. These files serve as experiment logs as well.

Training ThermoMPNN requires the use of a GPU. On a small dataset (<5000 data points), training takes <30s per epoch, while on a mega-scale dataset (>200,000 data points), it takes 8-12min per epoch (on a single V100 GPU). An example training SLURM script is provided at ```examples/train.sh```.

### Splits and Model Weights
For the purpose of replication and future benchmarking, the dataset splits used in this study are included as ```.pkl``` files under the ```dataset_splits/``` directory.

ThermoMPNN model weights can be found in the ```models/``` directory. The following model weights are provided:
```
- thermoMPNN_default.pt (best ThermoMPNN model trained on Megascale training dataset)
```

### Dataset Availability
The datasets used in this study can be found in the following locations:

Fireprot: https://doi.org/10.5281/zenodo.8169288
Megascale: https://doi.org/10.5281/zenodo.7401274
S669: https://doi.org/10.1093/bib/bbab555
SSYM, P53, Myoglobin, etc: https://protddg-bench.github.io
