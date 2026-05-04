from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nbt import NbtByte, NbtList, NbtShort, write_named_compound


@dataclass(frozen=True, slots=True)
class LegacyBlock:
    block_id: int
    data: int = 0


LEGACY_BLOCK_MAP: dict[str, LegacyBlock] = {
    "minecraft:air": LegacyBlock(0, 0),
    "minecraft:stone": LegacyBlock(1, 0),
    "minecraft:grass": LegacyBlock(2, 0),
    "minecraft:dirt": LegacyBlock(3, 0),
    "minecraft:cobblestone": LegacyBlock(4, 0),
    "minecraft:planks:0": LegacyBlock(5, 0),
    "minecraft:oak_planks": LegacyBlock(5, 0),
    "minecraft:spruce_planks": LegacyBlock(5, 1),
    "minecraft:water": LegacyBlock(9, 0),
    "minecraft:sand": LegacyBlock(12, 0),
    "minecraft:gravel": LegacyBlock(13, 0),
    "minecraft:log:0": LegacyBlock(17, 0),
    "minecraft:log:1": LegacyBlock(17, 1),
    "minecraft:leaves:0": LegacyBlock(18, 0),
    "minecraft:leaves:1": LegacyBlock(18, 1),
    "minecraft:leaves:3": LegacyBlock(18, 3),
    "minecraft:sponge": LegacyBlock(19, 0),
    "minecraft:glass": LegacyBlock(20, 0),
    "minecraft:sandstone": LegacyBlock(24, 0),
    "minecraft:wool:0": LegacyBlock(35, 0),
    "minecraft:wool:7": LegacyBlock(35, 7),
    "minecraft:wool:13": LegacyBlock(35, 13),
    "minecraft:obsidian": LegacyBlock(49, 0),
    "minecraft:snow_block": LegacyBlock(80, 0),
    "minecraft:quartz_block": LegacyBlock(155, 0),
    "minecraft:stained_glass:0": LegacyBlock(95, 0),
    "minecraft:stained_glass:3": LegacyBlock(95, 3),
    "minecraft:stained_glass:11": LegacyBlock(95, 11),
    "minecraft:stonebrick": LegacyBlock(98, 0),
    "minecraft:leaves2:1": LegacyBlock(161, 1),
    "minecraft:stained_hardened_clay:0": LegacyBlock(159, 0),
    "minecraft:stained_hardened_clay:12": LegacyBlock(159, 12),
}


@dataclass(slots=True)
class SchematicVolume:
    width: int
    height: int
    length: int
    block_states: np.ndarray


class SchematicWriter:
    def write(self, volume: SchematicVolume, path: Path) -> None:
        blocks, data, add_blocks = self._encode_legacy_arrays(volume.block_states)
        schematic = {
            "Width": NbtShort(volume.width),
            "Height": NbtShort(volume.height),
            "Length": NbtShort(volume.length),
            "Materials": "Alpha",
            "Blocks": blocks,
            "Data": data,
            "Entities": NbtList(element_type=10, values=[]),
            "TileEntities": NbtList(element_type=10, values=[]),
        }
        if add_blocks:
            schematic["AddBlocks"] = add_blocks
        write_named_compound(path, "Schematic", schematic)

    def _encode_legacy_arrays(self, block_states: np.ndarray) -> tuple[bytes, bytes, bytes]:
        # Legacy .schematic expects x as the fastest axis, then z, then y:
        # index = x + z * Width + y * Width * Length.
        # Our in-memory array is shaped as (x, y, z), so transpose to (x, z, y)
        # before Fortran-order flattening to preserve the expected storage order.
        flattened = block_states.transpose(0, 2, 1).ravel(order="F").tolist()
        blocks = bytearray()
        data = bytearray()
        add_blocks_nibbles: list[int] = []

        for block_state in flattened:
            legacy = LEGACY_BLOCK_MAP.get(block_state)
            if legacy is None:
                legacy = LEGACY_BLOCK_MAP["minecraft:stone"]
            blocks.append(legacy.block_id & 0xFF)
            data.append(legacy.data & 0x0F)
            add_blocks_nibbles.append((legacy.block_id >> 8) & 0x0F)

        add_blocks = bytearray()
        for index in range(0, len(add_blocks_nibbles), 2):
            low = add_blocks_nibbles[index]
            high = add_blocks_nibbles[index + 1] if index + 1 < len(add_blocks_nibbles) else 0
            add_blocks.append((high << 4) | low)

        if any(add_blocks):
            add_blocks_bytes = bytes(add_blocks)
        else:
            add_blocks_bytes = b""

        return bytes(blocks), bytes(data), add_blocks_bytes
