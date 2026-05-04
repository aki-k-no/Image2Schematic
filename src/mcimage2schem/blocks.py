from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True, slots=True)
class BlockCandidate:
    block_state: str
    rgb: tuple[int, int, int]
    weight: int = 1


GENERIC_COLOR_CANDIDATES: list[BlockCandidate] = [
    BlockCandidate("minecraft:stone", (125, 125, 125), 2),
    BlockCandidate("minecraft:cobblestone", (117, 117, 117), 2),
    BlockCandidate("minecraft:stonebrick", (136, 136, 136), 2),
    BlockCandidate("minecraft:dirt", (134, 96, 67), 2),
    BlockCandidate("minecraft:grass", (109, 153, 48), 2),
    BlockCandidate("minecraft:sand", (219, 211, 160), 2),
    BlockCandidate("minecraft:sandstone", (216, 202, 155), 2),
    BlockCandidate("minecraft:snow_block", (249, 254, 254), 2),
    BlockCandidate("minecraft:wool:0", (234, 236, 237), 2),
    BlockCandidate("minecraft:wool:7", (85, 85, 85), 2),
    BlockCandidate("minecraft:wool:13", (86, 99, 50), 2),
    BlockCandidate("minecraft:wool:5", (105, 118, 53), 2),
    BlockCandidate("minecraft:wool:3", (114, 136, 145), 2),
    BlockCandidate("minecraft:wool:11", (47, 59, 141), 2),
    BlockCandidate("minecraft:stained_hardened_clay:0", (209, 178, 161), 2),
    BlockCandidate("minecraft:stained_hardened_clay:4", (185, 133, 36), 2),
    BlockCandidate("minecraft:stained_hardened_clay:5", (103, 117, 52), 2),
    BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
    BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 2),
    BlockCandidate("minecraft:stained_hardened_clay:13", (74, 92, 39), 2),
    BlockCandidate("minecraft:oak_planks", (162, 130, 79), 2),
    BlockCandidate("minecraft:spruce_planks", (114, 84, 54), 2),
    BlockCandidate("minecraft:birch_planks", (193, 176, 118), 2),
    BlockCandidate("minecraft:log:0", (102, 81, 51), 2),
    BlockCandidate("minecraft:log:1", (81, 65, 44), 2),
    BlockCandidate("minecraft:leaves:0", (73, 112, 55), 2),
    BlockCandidate("minecraft:leaves:1", (84, 118, 63), 2),
    BlockCandidate("minecraft:leaves:3", (58, 90, 52), 2),
    BlockCandidate("minecraft:leaves2:1", (89, 140, 61), 2),
    BlockCandidate("minecraft:quartz_block", (221, 221, 221), 2),
]


PALETTE: dict[str, list[BlockCandidate]] = {
    "sky": [
        BlockCandidate("minecraft:stained_glass:3", (117, 204, 255), 5),
        BlockCandidate("minecraft:stained_glass:0", (240, 240, 255), 3),
        BlockCandidate("minecraft:wool:3", (114, 136, 145), 1),
    ],
    "cloud": [
        BlockCandidate("minecraft:wool:0", (236, 236, 236), 7),
        BlockCandidate("minecraft:stained_glass:0", (240, 240, 255), 3),
        BlockCandidate("minecraft:quartz_block", (221, 221, 221), 2),
    ],
    "water": [
        BlockCandidate("minecraft:water", (63, 118, 228), 7),
        BlockCandidate("minecraft:stained_glass:11", (60, 68, 170), 3),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 2),
        BlockCandidate("minecraft:wool:11", (47, 59, 141), 2),
        BlockCandidate("minecraft:wool:3", (114, 136, 145), 1),
    ],
    "water_deep": [
        BlockCandidate("minecraft:water", (40, 88, 170), 8),
        BlockCandidate("minecraft:stained_glass:11", (60, 68, 170), 5),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 2),
        BlockCandidate("minecraft:wool:11", (47, 59, 141), 3),
    ],
    "water_shallow": [
        BlockCandidate("minecraft:water", (92, 145, 210), 7),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 5),
        BlockCandidate("minecraft:stained_glass:0", (190, 220, 230), 2),
        BlockCandidate("minecraft:wool:3", (114, 136, 145), 2),
    ],
    "foliage": [
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 11),
        BlockCandidate("minecraft:leaves:1", (84, 118, 63), 9),
        BlockCandidate("minecraft:leaves:3", (58, 90, 52), 6),
        BlockCandidate("minecraft:leaves2:1", (89, 140, 61), 10),
        BlockCandidate("minecraft:wool:13", (86, 99, 50), 4),
        BlockCandidate("minecraft:wool:5", (105, 118, 53), 4),
        BlockCandidate("minecraft:stained_hardened_clay:13", (74, 92, 39), 3),
        BlockCandidate("minecraft:grass", (95, 159, 53), 2),
        BlockCandidate("minecraft:log:1", (81, 65, 44), 2),
        BlockCandidate("minecraft:log:0", (102, 81, 51), 2),
    ],
    "foliage_dark": [
        BlockCandidate("minecraft:leaves:3", (58, 90, 52), 10),
        BlockCandidate("minecraft:leaves:1", (84, 118, 63), 8),
        BlockCandidate("minecraft:log:1", (81, 65, 44), 4),
        BlockCandidate("minecraft:wool:13", (86, 99, 50), 3),
        BlockCandidate("minecraft:stained_hardened_clay:13", (74, 92, 39), 3),
        BlockCandidate("minecraft:wool:7", (85, 85, 85), 1),
    ],
    "foliage_light": [
        BlockCandidate("minecraft:leaves2:1", (89, 140, 61), 10),
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 7),
        BlockCandidate("minecraft:grass", (95, 159, 53), 4),
        BlockCandidate("minecraft:log:0", (102, 81, 51), 2),
        BlockCandidate("minecraft:wool:5", (105, 118, 53), 3),
        BlockCandidate("minecraft:stained_hardened_clay:5", (103, 117, 52), 2),
    ],
    "snow": [
        BlockCandidate("minecraft:snow_block", (249, 254, 254), 8),
        BlockCandidate("minecraft:wool:0", (234, 236, 237), 4),
        BlockCandidate("minecraft:quartz_block", (221, 221, 221), 2),
        BlockCandidate("minecraft:stained_hardened_clay:0", (209, 178, 161), 1),
    ],
    "sand": [
        BlockCandidate("minecraft:sandstone", (216, 202, 155), 8),
        BlockCandidate("minecraft:sand", (219, 211, 160), 6),
        BlockCandidate("minecraft:stained_hardened_clay:0", (209, 178, 161), 3),
        BlockCandidate("minecraft:birch_planks", (193, 176, 118), 1),
        BlockCandidate("minecraft:stained_hardened_clay:4", (185, 133, 36), 1),
    ],
    "rock": [
        BlockCandidate("minecraft:stone", (125, 125, 125), 9),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 5),
        BlockCandidate("minecraft:stonebrick", (136, 136, 136), 6),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
        BlockCandidate("minecraft:wool:7", (85, 85, 85), 1),
    ],
    "rock_dark": [
        BlockCandidate("minecraft:cobblestone", (100, 100, 100), 8),
        BlockCandidate("minecraft:stone", (110, 110, 110), 6),
        BlockCandidate("minecraft:obsidian", (76, 76, 79), 2),
        BlockCandidate("minecraft:wool:7", (85, 85, 85), 3),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 1),
    ],
    "cliff": [
        BlockCandidate("minecraft:stone", (125, 125, 125), 7),
        BlockCandidate("minecraft:stonebrick", (136, 136, 136), 5),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 5),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 1),
    ],
    "wood": [
        BlockCandidate("minecraft:oak_planks", (162, 130, 79), 5),
        BlockCandidate("minecraft:log:0", (102, 81, 51), 5),
        BlockCandidate("minecraft:log:1", (115, 85, 49), 4),
        BlockCandidate("minecraft:spruce_planks", (114, 84, 54), 4),
        BlockCandidate("minecraft:birch_planks", (193, 176, 118), 2),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 1),
    ],
    "shadow": [
        BlockCandidate("minecraft:wool:7", (55, 58, 62), 6),
        BlockCandidate("minecraft:obsidian", (76, 76, 79), 2),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 1),
    ],
    "ground": [
        BlockCandidate("minecraft:dirt", (134, 96, 67), 7),
        BlockCandidate("minecraft:grass", (109, 153, 48), 6),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 4),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
        BlockCandidate("minecraft:stone", (125, 125, 125), 1),
    ],
    "ground_grass": [
        BlockCandidate("minecraft:grass", (109, 153, 48), 9),
        BlockCandidate("minecraft:dirt", (134, 96, 67), 5),
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 2),
        BlockCandidate("minecraft:wool:5", (105, 118, 53), 3),
        BlockCandidate("minecraft:stained_hardened_clay:5", (103, 117, 52), 2),
    ],
    "ground_dirt": [
        BlockCandidate("minecraft:dirt", (134, 96, 67), 9),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 6),
        BlockCandidate("minecraft:stone", (125, 125, 125), 2),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
    ],
    "path": [
        BlockCandidate("minecraft:gravel", (134, 134, 134), 7),
        BlockCandidate("minecraft:dirt", (134, 96, 67), 6),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 4),
        BlockCandidate("minecraft:stained_hardened_clay:8", (134, 107, 98), 2),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 1),
    ],
}


class BlockSelector:
    def choose(
        self,
        label: str,
        rgb: tuple[int, int, int],
        default_block: str,
        image_x: int = 0,
        image_y: int = 0,
        voxel_x: int = 0,
        voxel_y: int = 0,
        voxel_z: int = 0,
        depth_value: float = 0.0,
    ) -> str:
        candidates = PALETTE.get(label)
        if not candidates:
            candidates = []

        generic = self._generic_candidates_for_label(label)
        merged_candidates = self._merge_candidates(candidates + generic)
        if not merged_candidates:
            return default_block
        if len(merged_candidates) == 1:
            return merged_candidates[0].block_state

        sorted_candidates = sorted(
            merged_candidates,
            key=lambda candidate: self._score_candidate(candidate, label, rgb),
        )
        best_score = self._score_candidate(sorted_candidates[0], label, rgb)
        tolerance = self._tolerance_for_label(label)
        eligible = [
            candidate
            for candidate in sorted_candidates
            if self._score_candidate(candidate, label, rgb) <= best_score + tolerance
        ]
        if not eligible:
            eligible = [sorted_candidates[0]]

        max_candidates = self._max_candidates_for_label(label)
        eligible = eligible[:max_candidates]
        if len(eligible) == 1:
            return eligible[0].block_state

        depth_bucket = int(floor(depth_value * 31.0))
        pick = self._palette_pick(
            image_x=image_x,
            image_y=image_y,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            voxel_z=voxel_z,
            depth_bucket=depth_bucket,
            palette_length=sum(max(candidate.weight, 1) for candidate in eligible),
        )

        running = 0
        for candidate in eligible:
            running += max(candidate.weight, 1)
            if pick < running:
                return candidate.block_state
        return eligible[-1].block_state

    @staticmethod
    def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
        return sum((l - r) ** 2 for l, r in zip(left, right))

    def _score_candidate(
        self,
        candidate: BlockCandidate,
        label: str,
        rgb: tuple[int, int, int],
    ) -> int:
        score = self._distance(candidate.rgb, rgb)
        if candidate not in PALETTE.get(label, []):
            score += self._generic_penalty_for_label(label)
        return score

    @staticmethod
    def _palette_pick(
        image_x: int,
        image_y: int,
        voxel_x: int,
        voxel_y: int,
        voxel_z: int,
        depth_bucket: int,
        palette_length: int,
    ) -> int:
        hashed = abs(
            (image_x * 31)
            + (image_y * 17)
            + (voxel_x * 19)
            + (voxel_y * 23)
            + (voxel_z * 13)
            + (depth_bucket * 29)
        )
        return hashed % max(palette_length, 1)

    @staticmethod
    def _tolerance_for_label(label: str) -> int:
        return {
            "foliage": 2800,
            "foliage_dark": 2400,
            "foliage_light": 2400,
            "ground": 2200,
            "ground_grass": 2000,
            "ground_dirt": 2000,
            "path": 1800,
            "rock": 1800,
            "rock_dark": 1600,
            "cliff": 1700,
            "sand": 1600,
            "snow": 1500,
            "water": 1300,
            "water_deep": 1100,
            "water_shallow": 1200,
            "cloud": 1000,
            "wood": 1700,
            "shadow": 1000,
        }.get(label, 1400)

    @staticmethod
    def _max_candidates_for_label(label: str) -> int:
        return {
            "foliage": 4,
            "foliage_dark": 4,
            "foliage_light": 4,
            "ground": 3,
            "ground_grass": 3,
            "ground_dirt": 3,
            "path": 3,
            "rock": 3,
            "rock_dark": 3,
            "cliff": 4,
            "sand": 3,
            "snow": 3,
            "water": 2,
            "water_deep": 2,
            "water_shallow": 3,
            "cloud": 2,
            "wood": 3,
            "shadow": 2,
        }.get(label, 2)

    @staticmethod
    def _generic_penalty_for_label(label: str) -> int:
        return {
            "foliage": 1100,
            "foliage_dark": 1200,
            "foliage_light": 1000,
            "ground": 900,
            "ground_grass": 900,
            "ground_dirt": 900,
            "path": 700,
            "rock": 900,
            "rock_dark": 800,
            "cliff": 900,
            "sand": 800,
            "snow": 700,
            "water": 1200,
            "water_deep": 1300,
            "water_shallow": 1100,
            "wood": 700,
            "shadow": 800,
        }.get(label, 1000)

    @staticmethod
    def _generic_candidates_for_label(label: str) -> list[BlockCandidate]:
        if label in {"sky", "cloud"}:
            return []
        if label in {"water", "water_deep", "water_shallow"}:
            allow = {
                "minecraft:stained_glass:11",
                "minecraft:stained_glass:3",
                "minecraft:stained_glass:0",
                "minecraft:wool:11",
                "minecraft:wool:3",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        if label in {"foliage", "foliage_dark", "foliage_light", "ground", "ground_grass", "ground_dirt", "path"}:
            allow = {
                "minecraft:dirt",
                "minecraft:grass",
                "minecraft:wool:13",
                "minecraft:wool:5",
                "minecraft:stained_hardened_clay:5",
                "minecraft:stained_hardened_clay:8",
                "minecraft:stained_hardened_clay:12",
                "minecraft:stained_hardened_clay:13",
                "minecraft:leaves:0",
                "minecraft:leaves:1",
                "minecraft:leaves:3",
                "minecraft:leaves2:1",
                "minecraft:log:0",
                "minecraft:log:1",
                "minecraft:cobblestone",
                "minecraft:stone",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        if label in {"rock", "rock_dark", "cliff", "shadow"}:
            allow = {
                "minecraft:stone",
                "minecraft:cobblestone",
                "minecraft:stonebrick",
                "minecraft:wool:7",
                "minecraft:obsidian",
                "minecraft:stained_hardened_clay:8",
                "minecraft:stained_hardened_clay:12",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        if label == "sand":
            allow = {
                "minecraft:sand",
                "minecraft:sandstone",
                "minecraft:stained_hardened_clay:0",
                "minecraft:stained_hardened_clay:4",
                "minecraft:birch_planks",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        if label == "snow":
            allow = {
                "minecraft:snow_block",
                "minecraft:wool:0",
                "minecraft:quartz_block",
                "minecraft:stained_hardened_clay:0",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        if label == "wood":
            allow = {
                "minecraft:oak_planks",
                "minecraft:spruce_planks",
                "minecraft:birch_planks",
                "minecraft:log:0",
                "minecraft:log:1",
                "minecraft:stained_hardened_clay:12",
            }
            return [candidate for candidate in GENERIC_COLOR_CANDIDATES if candidate.block_state in allow]
        return []

    @staticmethod
    def _merge_candidates(candidates: list[BlockCandidate]) -> list[BlockCandidate]:
        merged: dict[str, BlockCandidate] = {}
        for candidate in candidates:
            existing = merged.get(candidate.block_state)
            if existing is None:
                merged[candidate.block_state] = candidate
            else:
                merged[candidate.block_state] = BlockCandidate(
                    block_state=candidate.block_state,
                    rgb=candidate.rgb,
                    weight=max(existing.weight, candidate.weight),
                )
        return list(merged.values())
