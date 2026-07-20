#!/usr/bin/env python3
"""Regenerate cbmc_unknown_side_effect.goto.

The fixture pins ESBMC's graceful decline of a side-effect statement it has no
case for. `allocate` is used because it is the real one: JBMC's own lowering
rewrites java_new/java_new_array into side_effect_exprt(ID_allocate, ...), and
ESBMC's migrate.cpp dispatch recognises malloc/realloc/alloca/cpp_new/... but
not allocate (docs/jbmc-goto-binary-poc-plan.md 2.6). Phase 2 implements it;
until then the contract is a clean error rather than SIGABRT.

    ./gen_cbmc_unknown_side_effect.py

Regenerated with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

OUTPUT = "cbmc_unknown_side_effect.goto"

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}
SIZE = {"id": "unsignedbv", "namedSub": {"width": {"id": "64"}}}
VOID_PTR = {"id": "pointer",
            "namedSub": {"width": {"id": "64"}},
            "sub": [{"id": "empty"}]}

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


def main():
    p = {"id": "symbol",
         "namedSub": {"identifier": {"id": "p"}, "type": VOID_PTR}}

    # side_effect_exprt(ID_allocate, {size, zero-init}, void*) -- the shape
    # remove_java_new.cpp:100-105 emits for an object allocation.
    allocate = {
        "id": "side_effect",
        "namedSub": {"statement": {"id": "allocate"}, "type": VOID_PTR},
        "sub": [constant("8", SIZE), constant("0", INT)],
    }

    body = {
        "id": "code",
        "namedSub": {"statement": {"id": "block"}, "type": {"id": "code"}},
        "sub": [{
            "id": "code",
            "namedSub": {"statement": {"id": "assign"}, "type": {"id": "code"}},
            "sub": [p, allocate],
        }],
    }

    table = {
        "p": symbol("p", isStaticLifetime=True, isLvalue=True, type=VOID_PTR),
        "main": symbol(
            "main",
            type={"id": "code",
                  "namedSub": {"return_type": INT, "parameters": {"id": ""}}},
            value=body),
    }

    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        symtab = os.path.join(tmp, "symtab.json")
        with open(symtab, "w", encoding="utf-8") as out:
            json.dump({"symbolTable": table}, out)
        subprocess.run(
            ["symtab2gb", symtab, "--out", os.path.join(here, OUTPUT)],
            check=True)


if __name__ == "__main__":
    main()
