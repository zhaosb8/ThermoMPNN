from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geef_adapter import ThermoMPNNConfig, ThermoMPNNScorer, VariantInput


def main() -> None:
    pdb_candidates = sorted((Path("/home/zhaosb/GEEF/target_pdbs")).glob("*.pdb"))
    if not pdb_candidates:
        raise FileNotFoundError("No PDB found under /home/zhaosb/GEEF/target_pdbs")

    pdb_path = str(pdb_candidates[0])
    config = ThermoMPNNConfig(
        model_path=str(repo_root / "models" / "thermoMPNN_default.pt"),
        local_yaml_path=str(repo_root / "local.yaml"),
        device="cuda",
        default_chain_id="A",
    )
    scorer = ThermoMPNNScorer(config)

    wildtype_sequence = "MSTAGKVIKCKAAVAWEAGKPLSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTPLPNAQNVSVVDLTVRSLGADVVVVATGRARQGADVVVV"
    variant_sequence = list(wildtype_sequence)
    variant_sequence[4] = "R"
    variant_sequence[9] = "A"
    variant_sequence = "".join(variant_sequence)

    variant = VariantInput(
        variant_id="demo_variant_001",
        wildtype_sequence=wildtype_sequence,
        variant_sequence=variant_sequence,
        pdb_path=pdb_path,
        chain_id="A",
        mutation_list=["G5R", "C10A"],
        metadata={"source": "geef_adapter_smoke_test"},
    )

    result = scorer.score_variant(variant)
    print(json.dumps({
        "variant_id": result.variant_id,
        "status": result.status,
        "ddg_sum": result.ddg_sum,
        "mutation_list": result.mutation_list,
        "per_mutation_scores": [
            {"mutation": item.mutation, "ddg": item.ddg} for item in result.per_mutation_scores
        ],
        "error_type": result.error_type,
        "error_message": result.error_message,
    }, indent=2))


if __name__ == "__main__":
    main()
