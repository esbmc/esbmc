#!/usr/bin/env python3
"""Regenerate cbmc_unknown_side_effect.goto.

The fixture pins ESBMC's graceful decline of a side-effect statement it has no
case for.

It originally used `allocate`, the real blocker from
docs/jbmc-goto-binary-poc-plan.md 2.6, and served as the negative test that
Phase 2 would have to update. Phase 2 has since implemented `allocate`, so the
statement here is a synthetic id instead: the contract under test is the
graceful decline itself, which must keep working for whatever construct is
unsupported next.

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

    allocate = {
        "id": "side_effect",
        "namedSub": {"statement": {"id": "totally_unknown_side_effect"},
                     "type": VOID_PTR},
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
