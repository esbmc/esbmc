#!/usr/bin/env python3
"""Regenerate cbmc_allocate{,_fail}.goto.

Pins ESBMC's mapping of CBMC's generic `allocate` side effect onto malloc
(docs/jbmc-goto-binary-poc-plan.md 2.6). `allocate` is what remove_java_new
lowers java_new into, but it is ordinary CPROVER vocabulary that any
goto-instrument-lowered binary may carry, so it is worth pinning on its own.

The pass fixture allocates, writes through the returned pointer and reads the
value back. The fail fixture asserts the read-back value is something else, and
must report VERIFICATION FAILED -- without it a mapping that dropped the store
entirely, or returned a pointer to nondet memory, could still satisfy the pass
fixture if the assertion happened to be vacuous.

    ./gen_cbmc_allocate.py

Regenerated with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}
SIZE = {"id": "unsignedbv", "namedSub": {"width": {"id": "64"}}}
BOOL = {"id": "bool"}
INT_PTR = {"id": "pointer", "namedSub": {"width": {"id": "64"}}, "sub": [INT]}

FLAGS = [
    "isAuxiliary", "isExported", "isExtern", "isFileLocal", "isInput",
    "isLvalue", "isMacro", "isOutput", "isParameter", "isProperty",
    "isStateVar", "isStaticLifetime", "isThreadLocal", "isType", "isVolatile",
    "isWeak",
]


def symbol(name, **kw):
    sym: Dict[str, Any] = {flag: False for flag in FLAGS}
    sym.update({
        "baseName": name, "location": {}, "mode": "C", "module": "m",
        "name": name, "prettyName": name, "prettyType": "", "prettyValue": "",
        "type": {"id": "empty"}, "value": {"id": "nil"},
    })
    sym.update(kw)
    return sym


def constant(value, type_):
    return {"id": "constant",
            "namedSub": {"value": {"id": value}, "type": type_}}


def sym_expr(name, type_):
    return {"id": "symbol",
            "namedSub": {"identifier": {"id": name}, "type": type_}}


def deref(ptr, type_):
    return {"id": "dereference", "namedSub": {"type": type_}, "sub": [ptr]}


def code(statement, operands, type_id="code"):
    return {
        "id": "code",
        "namedSub": {"statement": {"id": statement}, "type": {"id": type_id}},
        "sub": operands,
    }


def build(path, expected):
    p = sym_expr("p", INT_PTR)

    # side_effect_exprt(ID_allocate, {byte_size, zero_init}, int*) -- the shape
    # remove_java_new.cpp:99-103 emits.
    allocate = {
        "id": "side_effect",
        "namedSub": {"statement": {"id": "allocate"}, "type": INT_PTR},
        "sub": [constant("4", SIZE),
                {"id": "constant",
                 "namedSub": {"value": {"id": "false"}, "type": BOOL}}],
    }

    body = code("block", [
        code("assign", [p, allocate]),
        code("assign", [deref(p, INT), constant("2a", INT)]),
        code("assert", [{
            "id": "=",
            "namedSub": {"type": BOOL},
            "sub": [deref(p, INT), constant(expected, INT)],
        }]),
    ])

    table = {
        "p": symbol("p", isStaticLifetime=True, isLvalue=True, type=INT_PTR),
        "main": symbol(
            "main",
            type={"id": "code",
                  "namedSub": {"return_type": INT, "parameters": {"id": ""}}},
            value=body),
    }
    with tempfile.TemporaryDirectory() as tmp:
        symtab = os.path.join(tmp, "symtab.json")
        with open(symtab, "w", encoding="utf-8") as out:
            json.dump({"symbolTable": table}, out)
        subprocess.run(["symtab2gb", symtab, "--out", path], check=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # 0x2a is written; the fail variant expects 0x2b instead.
    build(os.path.join(here, "cbmc_allocate.goto"), "2a")
    build(
        os.path.join(here, os.pardir, "cbmc_allocate_fail",
                     "cbmc_allocate_fail.goto"), "2b")


if __name__ == "__main__":
    main()
