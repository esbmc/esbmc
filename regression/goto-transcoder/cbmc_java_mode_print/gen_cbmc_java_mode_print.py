#!/usr/bin/env python3
"""Regenerate cbmc_java_mode_print.goto.

A goto binary can carry symbols whose mode names a language ESBMC has no
frontend for -- CPROVER's `java`, which every jbmc-produced binary uses. Any
from_expr/from_type on such a symbol (--goto-functions-only, --show-vcc, a
counterexample) used to build a null languaget and abort, so this fixture is a
mode=java symbol table with nothing else Java-specific in it.

    ./gen_cbmc_java_mode_print.py

Regenerated with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}

FLAGS = [
    "isAuxiliary", "isExported", "isExtern", "isFileLocal", "isInput",
    "isLvalue", "isMacro", "isOutput", "isParameter", "isProperty",
    "isStateVar", "isStaticLifetime", "isThreadLocal", "isType", "isVolatile",
    "isWeak",
]


def symbol(name, **kw):
    sym: Dict[str, Any] = {flag: False for flag in FLAGS}
    sym.update({
        "baseName": name, "location": {}, "mode": "java", "module": "m",
        "name": name, "prettyName": name, "prettyType": "", "prettyValue": "",
        "type": {"id": "empty"}, "value": {"id": "nil"},
    })
    sym.update(kw)
    return sym


def constant(value):
    return {"id": "constant",
            "namedSub": {"value": {"id": value}, "type": INT}}


def assign(lhs, rhs):
    return {
        "id": "code",
        "namedSub": {"statement": {"id": "assign"}, "type": {"id": "code"}},
        "sub": [lhs, rhs],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    local = "main::1::x"
    body = {
        "id": "code",
        "namedSub": {"statement": {"id": "block"}, "type": {"id": "code"}},
        "sub": [assign({"id": "symbol",
                        "namedSub": {"identifier": {"id": local},
                                     "type": INT}},
                       constant("1"))],
    }
    table = {
        local: symbol(local, type=INT, isLvalue=True),
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
        subprocess.run(
            ["symtab2gb", symtab,
             "--out", os.path.join(here, "cbmc_java_mode_print.goto")],
            check=True)


if __name__ == "__main__":
    main()
