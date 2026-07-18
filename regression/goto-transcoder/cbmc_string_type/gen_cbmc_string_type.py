#!/usr/bin/env python3
"""Regenerate cbmc_string_type.goto.

The fixture pins ESBMC's graceful decline of CPROVER's symbolic `string` type,
which reaches the adapter from every JBMC binary via java.lang.Object's
@class_identifier component (docs/jbmc-goto-binary-poc-plan.md §2.3.1).

A real JBMC binary would be ~768 KiB and need a JDK to produce. Emitting the
symbol table directly keeps the fixture at ~550 bytes and regenerable from any
machine with cbmc's symtab2gb on PATH -- no JDK and no Java source involved.

    ./gen_cbmc_string_type.py && symtab2gb symtab.json --out cbmc_string_type.goto

Produced with cbmc 6.8.0 (GOTO_BINARY_VERSION 6).
"""

import json
from typing import Any, Dict

INT = {"id": "signedbv", "namedSub": {"width": {"id": "32"}}}
STRUCT_TAG = {"id": "struct_tag", "namedSub": {"identifier": {"id": "tag-S"}}}

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


def component(name, type_):
    return {
        "id": "",
        "namedSub": {
            "name": {"id": name},
            "#pretty_name": {"id": name},
            "type": type_,
        },
    }


def main():
    # Mirrors java.lang.Object's shape: a @class_identifier of CPROVER's
    # `string` type alongside an ordinary integer member.
    struct = {
        "id": "struct",
        "namedSub": {
            "tag": {"id": "S"},
            "components": {
                "id": "",
                "sub": [
                    component("@class_identifier", {"id": "string"}),
                    component("n", INT),
                ],
            },
        },
    }

    g = {"id": "symbol",
         "namedSub": {"identifier": {"id": "g"}, "type": STRUCT_TAG}}

    # The type must be reached through an instruction: a struct type symbol
    # that no instruction mentions is never handed to the adapter's type pass,
    # and ESBMC exits successfully without ever seeing the `string`.
    body = {
        "id": "code",
        "namedSub": {"statement": {"id": "block"}, "type": {"id": "code"}},
        "sub": [{
            "id": "code",
            "namedSub": {"statement": {"id": "assign"}, "type": {"id": "code"}},
            "sub": [g, g],
        }],
    }

    table = {
        "tag-S": symbol("tag-S", isType=True, type=struct),
        "g": symbol("g", isStaticLifetime=True, isLvalue=True,
                    type=STRUCT_TAG),
        "main": symbol(
            "main",
            type={"id": "code",
                  "namedSub": {"return_type": INT, "parameters": {"id": ""}}},
            value=body),
    }

    with open("symtab.json", "w", encoding="utf-8") as out:
        json.dump({"symbolTable": table}, out)


if __name__ == "__main__":
    main()
