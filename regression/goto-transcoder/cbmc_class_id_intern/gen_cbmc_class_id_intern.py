#!/usr/bin/env python3
"""Regenerate cbmc_class_id_intern{,_fail}.goto.

Pins the *semantics* of @class_identifier interning, not merely that it does
not crash (docs/jbmc-goto-binary-poc-plan.md 4.1.1). The adapter maps CPROVER's
bare `string` type to an integer tag and interns each distinct class-name
literal to a distinct value, which is faithful only if the map is injective and
program-wide:

    same literal     -> equal      (a per-call map would break this)
    distinct literal -> not equal  (a collision would break this)

The pass fixture asserts both. The fail fixture asserts that two *distinct*
literals compare equal, and must report VERIFICATION FAILED -- without it a
degenerate implementation that interned every literal to the same constant
would satisfy the pass fixture.

    ./gen_cbmc_class_id_intern.py

Regenerated with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}
BOOL = {"id": "bool"}
STRING = {"id": "string"}

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


def class_id(literal):
    """A class-identifier literal: a constant carrying the bare `string` type."""
    return {"id": "constant",
            "namedSub": {"value": {"id": literal}, "type": STRING}}


def compare(op, lhs, rhs):
    return {"id": op, "namedSub": {"type": BOOL}, "sub": [lhs, rhs]}


def assertion(cond):
    return {
        "id": "code",
        "namedSub": {"statement": {"id": "assert"}, "type": {"id": "code"}},
        "sub": [cond],
    }


def build(path, statements):
    body = {
        "id": "code",
        "namedSub": {"statement": {"id": "block"}, "type": {"id": "code"}},
        "sub": statements,
    }
    table = {
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
    a, b = "java::Alpha", "java::Beta"

    build(os.path.join(here, "cbmc_class_id_intern.goto"), [
        assertion(compare("=", class_id(a), class_id(a))),
        assertion(compare("notequal", class_id(a), class_id(b))),
    ])

    fail_dir = os.path.join(here, os.pardir, "cbmc_class_id_intern_fail")
    build(os.path.join(fail_dir, "cbmc_class_id_intern_fail.goto"), [
        assertion(compare("=", class_id(a), class_id(b))),
    ])


if __name__ == "__main__":
    main()
