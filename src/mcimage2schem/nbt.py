from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path


TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


@dataclass(slots=True)
class NbtList:
    element_type: int
    values: list


@dataclass(frozen=True, slots=True)
class NbtByte:
    value: int


@dataclass(frozen=True, slots=True)
class NbtShort:
    value: int


@dataclass(frozen=True, slots=True)
class NbtInt:
    value: int


def write_named_compound(path: Path, name: str, payload: dict) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(bytes([TAG_COMPOUND]))
        _write_string(handle, name)
        _write_compound_payload(handle, payload)


def _write_named_tag(handle, name: str, value) -> None:
    tag_type = _infer_tag_type(value)
    handle.write(bytes([tag_type]))
    _write_string(handle, name)
    _write_payload(handle, tag_type, value)


def _write_payload(handle, tag_type: int, value) -> None:
    if tag_type == TAG_BYTE:
        handle.write(struct.pack(">b", value.value if isinstance(value, NbtByte) else value))
    elif tag_type == TAG_SHORT:
        handle.write(struct.pack(">h", value.value if isinstance(value, NbtShort) else value))
    elif tag_type == TAG_INT:
        handle.write(struct.pack(">i", value.value if isinstance(value, NbtInt) else value))
    elif tag_type == TAG_STRING:
        _write_string(handle, value)
    elif tag_type == TAG_BYTE_ARRAY:
        handle.write(struct.pack(">i", len(value)))
        handle.write(bytes(value))
    elif tag_type == TAG_LIST:
        handle.write(bytes([value.element_type]))
        handle.write(struct.pack(">i", len(value.values)))
        for item in value.values:
            _write_payload(handle, value.element_type, item)
    elif tag_type == TAG_COMPOUND:
        _write_compound_payload(handle, value)
    elif tag_type == TAG_INT_ARRAY:
        handle.write(struct.pack(">i", len(value)))
        for item in value:
            handle.write(struct.pack(">i", item))
    else:
        raise TypeError(f"Unsupported NBT tag type: {tag_type}")


def _write_compound_payload(handle, payload: dict) -> None:
    for key, value in payload.items():
        _write_named_tag(handle, key, value)
    handle.write(bytes([TAG_END]))


def _write_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack(">h", len(encoded)))
    handle.write(encoded)


def _infer_tag_type(value) -> int:
    if isinstance(value, NbtList):
        return TAG_LIST
    if isinstance(value, NbtByte):
        return TAG_BYTE
    if isinstance(value, NbtShort):
        return TAG_SHORT
    if isinstance(value, NbtInt):
        return TAG_INT
    if isinstance(value, dict):
        return TAG_COMPOUND
    if isinstance(value, bytes):
        return TAG_BYTE_ARRAY
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, list):
        return TAG_INT_ARRAY
    if isinstance(value, int):
        if -128 <= value <= 127:
            return TAG_BYTE
        if -32768 <= value <= 32767:
            return TAG_SHORT
        return TAG_INT
    raise TypeError(f"Unsupported value for NBT serialization: {type(value)!r}")
