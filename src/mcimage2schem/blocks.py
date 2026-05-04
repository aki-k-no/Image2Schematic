from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True, slots=True)
class BlockCandidate:
    block_state: str
    rgb: tuple[int, int, int]
    weight: int = 1


PALETTE: dict[str, list[BlockCandidate]] = {
    "sky": [
        BlockCandidate("minecraft:stained_glass:3", (117, 204, 255), 5),
        BlockCandidate("minecraft:stained_glass:0", (240, 240, 255), 3),
    ],
    "cloud": [
        BlockCandidate("minecraft:wool:0", (236, 236, 236), 7),
        BlockCandidate("minecraft:stained_glass:0", (240, 240, 255), 3),
    ],
    "water": [
        BlockCandidate("minecraft:water", (63, 118, 228), 7),
        BlockCandidate("minecraft:stained_glass:11", (60, 68, 170), 3),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 2),
    ],
    "water_deep": [
        BlockCandidate("minecraft:water", (40, 88, 170), 8),
        BlockCandidate("minecraft:stained_glass:11", (60, 68, 170), 5),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 2),
    ],
    "water_shallow": [
        BlockCandidate("minecraft:water", (92, 145, 210), 7),
        BlockCandidate("minecraft:stained_glass:3", (97, 125, 142), 5),
        BlockCandidate("minecraft:stained_glass:0", (190, 220, 230), 2),
    ],
    "foliage": [
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 11),
        BlockCandidate("minecraft:leaves:1", (84, 118, 63), 9),
        BlockCandidate("minecraft:leaves:3", (58, 90, 52), 6),
        BlockCandidate("minecraft:leaves2:1", (89, 140, 61), 10),
        BlockCandidate("minecraft:wool:13", (86, 99, 50), 4),
        BlockCandidate("minecraft:grass", (95, 159, 53), 2),
        BlockCandidate("minecraft:log:1", (81, 65, 44), 2),
    ],
    "foliage_dark": [
        BlockCandidate("minecraft:leaves:3", (58, 90, 52), 10),
        BlockCandidate("minecraft:leaves:1", (84, 118, 63), 8),
        BlockCandidate("minecraft:log:1", (81, 65, 44), 4),
        BlockCandidate("minecraft:wool:13", (86, 99, 50), 3),
    ],
    "foliage_light": [
        BlockCandidate("minecraft:leaves2:1", (89, 140, 61), 10),
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 7),
        BlockCandidate("minecraft:grass", (95, 159, 53), 4),
        BlockCandidate("minecraft:log:0", (102, 81, 51), 2),
    ],
    "snow": [
        BlockCandidate("minecraft:snow_block", (249, 254, 254), 8),
        BlockCandidate("minecraft:wool:0", (234, 236, 237), 4),
        BlockCandidate("minecraft:quartz_block", (221, 221, 221), 2),
    ],
    "sand": [
        BlockCandidate("minecraft:sandstone", (216, 202, 155), 8),
        BlockCandidate("minecraft:sand", (219, 211, 160), 6),
        BlockCandidate("minecraft:stained_hardened_clay:0", (209, 178, 161), 3),
    ],
    "rock": [
        BlockCandidate("minecraft:stone", (125, 125, 125), 9),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 5),
        BlockCandidate("minecraft:stonebrick", (136, 136, 136), 6),
        BlockCandidate("minecraft:gravel", (134, 134, 134), 3),
    ],
    "rock_dark": [
        BlockCandidate("minecraft:cobblestone", (100, 100, 100), 8),
        BlockCandidate("minecraft:stone", (110, 110, 110), 6),
        BlockCandidate("minecraft:obsidian", (76, 76, 79), 2),
    ],
    "cliff": [
        BlockCandidate("minecraft:stone", (125, 125, 125), 7),
        BlockCandidate("minecraft:stonebrick", (136, 136, 136), 5),
        BlockCandidate("minecraft:cobblestone", (117, 117, 117), 5),
        BlockCandidate("minecraft:gravel", (134, 134, 134), 3),
    ],
    "wood": [
        BlockCandidate("minecraft:oak_planks", (162, 130, 79), 5),
        BlockCandidate("minecraft:log:0", (102, 81, 51), 5),
        BlockCandidate("minecraft:log:1", (115, 85, 49), 4),
        BlockCandidate("minecraft:spruce_planks", (114, 84, 54), 4),
    ],
    "shadow": [
        BlockCandidate("minecraft:wool:7", (55, 58, 62), 6),
        BlockCandidate("minecraft:obsidian", (76, 76, 79), 2),
    ],
    "ground": [
        BlockCandidate("minecraft:dirt", (134, 96, 67), 7),
        BlockCandidate("minecraft:grass", (109, 153, 48), 6),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 4),
        BlockCandidate("minecraft:gravel", (134, 134, 134), 2),
    ],
    "ground_grass": [
        BlockCandidate("minecraft:grass", (109, 153, 48), 9),
        BlockCandidate("minecraft:dirt", (134, 96, 67), 5),
        BlockCandidate("minecraft:leaves:0", (73, 112, 55), 2),
    ],
    "ground_dirt": [
        BlockCandidate("minecraft:dirt", (134, 96, 67), 9),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 6),
        BlockCandidate("minecraft:gravel", (134, 134, 134), 3),
    ],
    "path": [
        BlockCandidate("minecraft:gravel", (134, 134, 134), 7),
        BlockCandidate("minecraft:dirt", (134, 96, 67), 6),
        BlockCandidate("minecraft:stained_hardened_clay:12", (119, 70, 51), 4),
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
            return default_block
        if len(candidates) == 1:
            return candidates[0].block_state

        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: self._distance(candidate.rgb, rgb),
        )
        best_distance = self._distance(sorted_candidates[0].rgb, rgb)
        tolerance = self._tolerance_for_label(label)
        eligible = [
            candidate
            for candidate in sorted_candidates
            if self._distance(candidate.rgb, rgb) <= best_distance + tolerance
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
            "foliage": 3400,
            "foliage_dark": 2800,
            "foliage_light": 2800,
            "ground": 2600,
            "ground_grass": 2200,
            "ground_dirt": 2200,
            "path": 2000,
            "rock": 2200,
            "rock_dark": 1800,
            "cliff": 1800,
            "sand": 1800,
            "snow": 1600,
            "water": 1400,
            "water_deep": 1200,
            "water_shallow": 1200,
            "cloud": 1200,
            "wood": 1800,
            "shadow": 1200,
        }.get(label, 1600)

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
