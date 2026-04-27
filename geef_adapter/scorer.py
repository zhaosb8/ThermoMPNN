from __future__ import annotations

from pathlib import Path

from .model_loader import load_thermompnn_model
from .mutation_parser import (
    MutationParserError,
    build_mutation_objects,
    diff_sequences_to_mutation_strings,
    normalize_mutation_list,
    validate_explicit_vs_inferred_mutations,
    validate_mutation_strings_against_wt,
)
from .structure_cache import StructureCache
from .types import PerMutationScore, ThermoMPNNConfig, VariantInput, VariantScoreResult


class StructureLoadError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


class ThermoMPNNScorer:
    def __init__(self, config: ThermoMPNNConfig) -> None:
        self.config = config
        self.model = load_thermompnn_model(config)
        self.structure_cache = StructureCache(enabled=config.cache_structures)

    def _build_error_result(self, variant: VariantInput, exc: Exception, mutation_list: list[str] | None = None) -> VariantScoreResult:
        return VariantScoreResult(
            variant_id=variant.variant_id,
            status="error",
            ddg_sum=None,
            mutation_list=mutation_list if mutation_list is not None else (variant.mutation_list or []),
            per_mutation_scores=[],
            aggregation_method=self.config.aggregation_method,
            mutation_count=len(mutation_list if mutation_list is not None else (variant.mutation_list or [])),
            warnings=[],
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            metadata={**variant.metadata},
        )

    def _get_parsed_pdb(self, pdb_path: str, chain_id: str):
        if not Path(pdb_path).exists():
            raise StructureLoadError(f"PDB file not found: {pdb_path}")
        try:
            return self.structure_cache.get_or_load(pdb_path, chain_id)
        except Exception as exc:  # noqa: BLE001
            raise StructureLoadError(f"failed to load structure {pdb_path} chain {chain_id}: {exc}") from exc

    def _prepare_mutations(self, variant: VariantInput) -> list[str]:
        if self.config.strict_sequence_length_check and len(variant.wildtype_sequence) != len(variant.variant_sequence):
            raise MutationParserError("wildtype_sequence and variant_sequence must have the same length")

        inferred_mutations = diff_sequences_to_mutation_strings(variant.wildtype_sequence, variant.variant_sequence)
        normalized_inferred = normalize_mutation_list(inferred_mutations)

        if variant.mutation_list:
            validate_explicit_vs_inferred_mutations(variant.mutation_list, normalized_inferred)
        if self.config.strict_mutation_validation:
            validate_mutation_strings_against_wt(variant.wildtype_sequence, normalized_inferred)
        return normalized_inferred

    def _predict_single_mutations(self, parsed_pdb, mutation_strings: list[str]) -> list[PerMutationScore]:
        if not mutation_strings:
            return []
        pdb_name = str(parsed_pdb[0].get("name", "input"))
        mutation_objects = build_mutation_objects(mutation_strings, pdb_name=pdb_name)
        try:
            predictions, _ = self.model(parsed_pdb, mutation_objects)
        except Exception as exc:  # noqa: BLE001
            raise ModelInferenceError(f"ThermoMPNN inference failed: {exc}") from exc

        per_mutation_scores: list[PerMutationScore] = []
        for mutation, prediction in zip(mutation_strings, predictions):
            if prediction is None:
                continue
            ddg_value = prediction["ddG"]
            if hasattr(ddg_value, "detach"):
                ddg_float = float(ddg_value.detach().cpu().item())
            else:
                ddg_float = float(ddg_value)
            per_mutation_scores.append(PerMutationScore(mutation=mutation, ddg=ddg_float))
        return per_mutation_scores

    def _predict_single_mutations_batch(self, parsed_pdb, mutation_strings: list[str]) -> dict[str, PerMutationScore]:
        if not mutation_strings:
            return {}
        per_mutation_scores = self._predict_single_mutations(parsed_pdb, mutation_strings)
        return {score.mutation: score for score in per_mutation_scores}

    def _aggregate_ddg(self, per_mutation_scores: list[PerMutationScore]) -> float:
        if self.config.aggregation_method != "sum":
            raise ValueError(f"unsupported aggregation method: {self.config.aggregation_method}")
        return float(sum(score.ddg for score in per_mutation_scores))

    def score_variant(self, variant: VariantInput) -> VariantScoreResult:
        try:
            normalized_inferred = self._prepare_mutations(variant)
            parsed_pdb = self._get_parsed_pdb(variant.pdb_path, variant.chain_id or self.config.default_chain_id)
            per_mutation_scores = self._predict_single_mutations(parsed_pdb, normalized_inferred)
            ddg_sum = self._aggregate_ddg(per_mutation_scores)
            return VariantScoreResult(
                variant_id=variant.variant_id,
                status="ok",
                ddg_sum=ddg_sum,
                mutation_list=normalized_inferred,
                per_mutation_scores=per_mutation_scores,
                aggregation_method=self.config.aggregation_method,
                mutation_count=len(normalized_inferred),
                warnings=[],
                metadata={**variant.metadata},
            )
        except Exception as exc:  # noqa: BLE001
            return self._build_error_result(variant, exc)

    def score_variants(self, variants: list[VariantInput]) -> list[VariantScoreResult]:
        if not variants:
            return []

        prepared_variants: list[tuple[VariantInput, list[str], tuple[str, str]]] = []
        errors: list[VariantScoreResult] = []
        parsed_pdb_cache: dict[tuple[str, str], object] = {}

        for variant in variants:
            try:
                normalized_inferred = self._prepare_mutations(variant)
                chain_id = variant.chain_id or self.config.default_chain_id
                cache_key = (variant.pdb_path, chain_id)
                if cache_key not in parsed_pdb_cache:
                    parsed_pdb_cache[cache_key] = self._get_parsed_pdb(variant.pdb_path, chain_id)
                prepared_variants.append((variant, normalized_inferred, cache_key))
            except Exception as exc:  # noqa: BLE001
                errors.append(self._build_error_result(variant, exc))

        grouped: dict[tuple[str, str], list[tuple[VariantInput, list[str]]]] = {}
        for variant, mutation_strings, cache_key in prepared_variants:
            grouped.setdefault(cache_key, []).append((variant, mutation_strings))

        results: list[VariantScoreResult] = []
        for cache_key, group_items in grouped.items():
            parsed_pdb = parsed_pdb_cache[cache_key]
            if self.config.combine_variant_mutations:
                unique_mutations = sorted({mutation for _, mutation_strings in group_items for mutation in mutation_strings})
                try:
                    score_map = self._predict_single_mutations_batch(parsed_pdb, unique_mutations)
                except Exception as exc:  # noqa: BLE001
                    results.extend(self._build_error_result(variant, exc, mutation_strings) for variant, mutation_strings in group_items)
                    continue

                for variant, mutation_strings in group_items:
                    missing = [mutation for mutation in mutation_strings if mutation not in score_map]
                    if missing:
                        results.append(
                            self._build_error_result(
                                variant,
                                ModelInferenceError(f"missing single-mutation predictions: {missing}"),
                                mutation_strings,
                            )
                        )
                        continue
                    per_mutation_scores = [score_map[mutation] for mutation in mutation_strings]
                    ddg_sum = self._aggregate_ddg(per_mutation_scores)
                    results.append(
                        VariantScoreResult(
                            variant_id=variant.variant_id,
                            status="ok",
                            ddg_sum=ddg_sum,
                            mutation_list=mutation_strings,
                            per_mutation_scores=per_mutation_scores,
                            aggregation_method=self.config.aggregation_method,
                            mutation_count=len(mutation_strings),
                            warnings=[],
                            metadata={**variant.metadata},
                        )
                    )
                continue

            for variant, mutation_strings in group_items:
                try:
                    per_mutation_scores = self._predict_single_mutations(parsed_pdb, mutation_strings)
                    ddg_sum = self._aggregate_ddg(per_mutation_scores)
                    results.append(
                        VariantScoreResult(
                            variant_id=variant.variant_id,
                            status="ok",
                            ddg_sum=ddg_sum,
                            mutation_list=mutation_strings,
                            per_mutation_scores=per_mutation_scores,
                            aggregation_method=self.config.aggregation_method,
                            mutation_count=len(mutation_strings),
                            warnings=[],
                            metadata={**variant.metadata},
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(self._build_error_result(variant, exc, mutation_strings))

        result_by_id = {result.variant_id: result for result in [*results, *errors]}
        return [result_by_id[variant.variant_id] for variant in variants if variant.variant_id in result_by_id]
