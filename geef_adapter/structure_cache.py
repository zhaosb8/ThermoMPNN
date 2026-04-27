from __future__ import annotations

from pathlib import Path

from protein_mpnn_utils import alt_parse_PDB


class StructureCache:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[tuple[str, str], list[dict[str, object]]] = {}

    def get_or_load(self, pdb_path: str, chain_id: str) -> list[dict[str, object]]:
        key = (str(Path(pdb_path).resolve()), chain_id)
        if self.enabled and key in self._cache:
            return self._cache[key]
        parsed = alt_parse_PDB(pdb_path, chain_id)
        if self.enabled:
            self._cache[key] = parsed
        return parsed

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
