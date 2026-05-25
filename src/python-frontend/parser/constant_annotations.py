"""Constant-annotation helpers used by parser orchestration."""

from __future__ import annotations

import ast
import base64

__all__ = [
    "add_type_annotation",
    "annotate_constant_node",
    "encode_bytes",
    "tag_bignum_constants",
]

# Python ints are arbitrary precision; the JSON wire format used by the C++
# frontend stores them as numbers, which nlohmann::json silently truncates to
# double once they exceed uint64.
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _fits_int64(v: int) -> bool:
    return _INT64_MIN <= v <= _INT64_MAX


def _is_usub_unaryop(node: object) -> bool:
    return (isinstance(node, dict) and node.get("_type") == "UnaryOp"
            and isinstance(node.get("op"), dict) and node["op"].get("_type") == "USub")


def _tag_bignum_constants(node: object, in_usub_operand: bool = False) -> None:
    if isinstance(node, dict):
        if node.get("_type") == "Constant":
            v = node.get("value")
            if isinstance(v, int) and not isinstance(v, bool):
                effective = -v if in_usub_operand else v
                if not _fits_int64(effective):
                    node["_bigint"] = str(v)
                    node["value"] = None
            return
        is_usub = _is_usub_unaryop(node)
        for k, v in node.items():
            _tag_bignum_constants(v, in_usub_operand=(is_usub and k == "operand"))
    elif isinstance(node, list):
        for v in node:
            _tag_bignum_constants(v)


def encode_bytes(value: bytes) -> str:
    """Encode raw bytes as ASCII base64 for JSON transport."""
    return base64.b64encode(value).decode("ascii")


def annotate_constant_node(value_node: ast.AST) -> None:
    """Attach ESBMC-specific type metadata to supported Constant nodes."""
    if not isinstance(value_node, ast.Constant):
        return

    if isinstance(value_node.value, str):
        value_node.esbmc_type_annotation = "str"
    elif isinstance(value_node.value, bytes):
        value_node.esbmc_type_annotation = "bytes"
        value_node.encoded_bytes = encode_bytes(value_node.value)
    elif isinstance(value_node.value, complex):
        value_node.esbmc_type_annotation = "complex"
        value_node.real_value = value_node.value.real
        value_node.imag_value = value_node.value.imag


def add_type_annotation(node: ast.Assign) -> None:
    """Propagate annotation to assignment RHS constant values."""
    annotate_constant_node(node.value)


def tag_bignum_constants(node: object) -> None:
    """Public façade for bigint tagging in AST-JSON dictionaries."""
    _tag_bignum_constants(node)
