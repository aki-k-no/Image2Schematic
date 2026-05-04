from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockCandidate:
    block_state: str
    rgb: tuple[int, int, int]


PALETTE: dict[str, list[BlockCandidate]] = {
    "sky": [
        BlockCandidate("minecraft:stained_glass:3", (117, 204, 255)),
        BlockCandidate("minecraft:stained_glass:0", (240, 240, 255)),
    ],
    "water": [
        BlockCandidate("minecraft:water", (63, 118, 228)),
        BlockCandidate("minecraft:stained_glass:11", (60, 68, 170)),
    ],
    "foliage": [
        BlockCandidate("minecraft:wool:13", (85, 110, 27)),
        BlockCandidate("minecraft:leaves:0", (92, 113, 56)),
        BlockCandidate("minecraft:leaves2:1", (89, 140, 61)),
    ],
    "snow": [
        BlockCandidate("minecraft:snow_block", (249, 254, 254)),
        BlockCandidate("minecraft:wool:0", (234, 236, 237)),
    ],
    "sand": [
        BlockCandidate("minecraft:sandstone", (216, 202, 155)),
        BlockCandidate("minecraft:sand", (219, 211, 160)),
    ],
    "rock": [
        BlockCandidate("minecraft:stone", (125, 125, 125)),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117)),
        BlockCandidate("minecraft:stonebrick", (136, 136, 136)),
    ],
    "wood": [
        BlockCandidate("minecraft:oak_planks", (162, 130, 79)),
        BlockCandidate("minecraft:log:1", (115, 85, 49)),
    ],
    "shadow": [
        BlockCandidate("minecraft:wool:7", (55, 58, 62)),
        BlockCandidate("minecraft:obsidian", (76, 76, 79)),
    ],
    "ground": [
        BlockCandidate("minecraft:dirt", (134, 96, 67)),
        BlockCandidate("minecraft:grass", (109, 153, 48)),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51)),
    ],
}


class BlockSelector:
    def choose(self, label: str, rgb: tuple[int, int, int], default_block: str) -> str:
        candidates = PALETTE.get(label)
        if not candidates:
            return default_block
        return min(candidates, key=lambda candidate: self._distance(candidate.rgb, rgb)).block_state

    @staticmethod
    def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
        return sum((l - r) ** 2 for l, r in zip(left, right))
