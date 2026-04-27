from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ThermoMPNNConfig:
    model_path: str
    local_yaml_path: str
    device: str = "cuda"
    default_chain_id: str = "A"
    strict_sequence_length_check: bool = True
    strict_mutation_validation: bool = True
    aggregation_method: str = "sum"
    cache_structures: bool = True
    combine_variant_mutations: bool = True


@dataclass(slots=True)
class VariantInput:
    variant_id: str
    wildtype_sequence: str
    variant_sequence: str
    pdb_path: str
    chain_id: str
    mutation_list: list[str] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PerMutationScore:
    mutation: str
    ddg: float


@dataclass(slots=True)
class VariantScoreResult:
    variant_id: str
    status: str
    ddg_sum: float | None
    mutation_list: list[str]
    per_mutation_scores: list[PerMutationScore]
    aggregation_method: str
    mutation_count: int
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
