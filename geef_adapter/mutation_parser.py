from __future__ import annotations

import re

from datasets import Mutation

_MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class MutationParserError(ValueError):
    pass


class SequenceLengthMismatchError(MutationParserError):
    pass


class InvalidMutationFormatError(MutationParserError):
    pass


class MutationValidationError(MutationParserError):
    pass


class MutationSequenceMismatchError(MutationParserError):
    pass


def diff_sequences_to_mutation_strings(wildtype_sequence: str, variant_sequence: str) -> list[str]:
    if len(wildtype_sequence) != len(variant_sequence):
        raise SequenceLengthMismatchError("wildtype_sequence and variant_sequence must have the same length")
    mutations: list[str] = []
    for index, (wt_residue, variant_residue) in enumerate(zip(wildtype_sequence, variant_sequence), start=1):
        if wt_residue != variant_residue:
            mutations.append(f"{wt_residue}{index}{variant_residue}")
    return mutations


def parse_mutation_string(mutation: str) -> tuple[str, int, str]:
    cleaned = mutation.strip().upper()
    match = _MUTATION_PATTERN.match(cleaned)
    if not match:
        raise InvalidMutationFormatError(f"invalid mutation format: {mutation}")
    wildtype, position, mutant = match.groups()
    return wildtype, int(position), mutant


def normalize_mutation_list(mutation_list: list[str]) -> list[str]:
    unique: dict[tuple[int, str, str], str] = {}
    for mutation in mutation_list:
        wildtype, position, mutant = parse_mutation_string(mutation)
        unique[(position, wildtype, mutant)] = f"{wildtype}{position}{mutant}"
    return [unique[key] for key in sorted(unique.keys(), key=lambda item: item[0])]


def validate_mutation_strings_against_wt(wildtype_sequence: str, mutation_strings: list[str]) -> None:
    for mutation in mutation_strings:
        wildtype, position, _ = parse_mutation_string(mutation)
        index = position - 1
        if index < 0 or index >= len(wildtype_sequence):
            raise MutationValidationError(f"mutation position out of range: {mutation}")
        if wildtype_sequence[index] != wildtype:
            raise MutationValidationError(
                f"wildtype residue mismatch for {mutation}: expected {wildtype_sequence[index]} at position {position}"
            )


def validate_explicit_vs_inferred_mutations(explicit_mutations: list[str], inferred_mutations: list[str]) -> None:
    normalized_explicit = normalize_mutation_list(explicit_mutations)
    normalized_inferred = normalize_mutation_list(inferred_mutations)
    if normalized_explicit != normalized_inferred:
        raise MutationSequenceMismatchError(
            f"explicit mutation_list does not match sequence-derived mutations: {normalized_explicit} != {normalized_inferred}"
        )


def build_mutation_objects(mutation_strings: list[str], pdb_name: str) -> list[Mutation]:
    mutation_objects: list[Mutation] = []
    for mutation in mutation_strings:
        wildtype, position, mutant = parse_mutation_string(mutation)
        mutation_objects.append(
            Mutation(
                position=position - 1,
                wildtype=wildtype,
                mutation=mutant,
                ddG=None,
                pdb=pdb_name,
            )
        )
    return mutation_objects
