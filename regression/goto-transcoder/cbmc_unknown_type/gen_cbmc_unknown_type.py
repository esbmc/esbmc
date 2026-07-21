#!/usr/bin/env python3
"""Regenerate cbmc_unknown_type.goto.

The fixture pins ESBMC's graceful decline of a type id migrate_type0 has no
case for -- the abort site docs/jbmc-goto-binary-poc-plan.md 2.3.1 identified
as JBMC's real first blocker.

It uses a deliberately synthetic id rather than a real one because every
Java-reachable candidate is intercepted earlier: `string` by
reject_string_type (cbmc_adapter.cpp:863) and `struct_tag` by the adapter's
resolution pass. A synthetic id is the only way to reach migrate_type0's
fall-through itself.

    ./gen_cbmc_unknown_type.py

Regenerated with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

OUTPUT = "cbmc_unknown_type.goto"

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}
UNKNOWN = {"id": "totally_unknown_type_id"}

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


def main():
    # The type must be reached through an instruction: a symbol no instruction
    # mentions is never handed to the type pass (see cbmc_string_type's gen).
    u = {"id": "symbol",
         "namedSub": {"identifier": {"id": "u"}, "type": UNKNOWN}}

    body = {
        "id": "code",
        "namedSub": {"statement": {"id": "block"}, "type": {"id": "code"}},
        "sub": [{
            "id": "code",
            "namedSub": {"statement": {"id": "assign"}, "type": {"id": "code"}},
            "sub": [u, u],
        }],
    }

    table = {
        "u": symbol("u", isStaticLifetime=True, isLvalue=True, type=UNKNOWN),
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
