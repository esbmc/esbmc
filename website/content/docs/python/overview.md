---
title: Overview
weight: 1
---

The Python frontend converts Python source files into ESBMC's internal representation (IRep), enabling the engine's bounded model checker and SMT backend to verify program properties.

> The pipeline has three stages: **AST generation** (Python) → **type annotation** (Python) → **symbol table generation** (C++). Each stage feeds into the next before ESBMC's backend takes over.

![ESBMC Python Frontend Architecture](https://raw.githubusercontent.com/esbmc/esbmc/master/src/python-frontend/images/arch.png)

## AST Generation

Python code translation starts by parsing `.py` files into an Abstract Syntax Tree (AST). This is done using Python's built-in [`ast`](https://docs.python.org/3/library/ast.html) module and the [`ast2json`](https://pypi.org/project/ast2json/) package, which serializes the AST to JSON. The process runs alongside the Python interpreter and produces a JSON file for each `.py` file processed, including any imported modules.

This approach's main advantage is that it relies on a native Python module, ensuring the parsed representation faithfully reflects the language.

The parser is implemented as the `src/python-frontend/parser/` package. ESBMC invokes its `parser/__main__.py` entrypoint with a Python 3 interpreter; module discovery, cycle detection, and relative-import rewriting live in `parser/import_resolver.py`, which emits structured diagnostics on missing modules and cyclic imports.

## Type Annotation

After generating the AST, the frontend traverses the JSON tree and inserts additional nodes carrying type information. [PEP 484](https://peps.python.org/pep-0484/) introduced an optional type system, allowing developers to annotate variables using the `var_name: type` syntax.

Where explicit annotations are absent, the frontend infers types from constants, previously annotated variables, binary expressions, and class instances.

The JSON below shows the annotated representation of `x: int = 10`:

```json
{
    "_type": "AnnAssign",
    "annotation": {
        "_type": "Name",
        "id": "int"
    },
    "target": {
        "_type": "Name",
        "id": "x"
    },
    "value": {
        "_type": "Constant",
        "value": 10
    }
}
```

## Symbol Table Generation

The final frontend step converts the annotated JSON AST into a symbol table using ESBMC's C++ IRep API. This API builds a control-flow graph (CFG) from the program, modelling assignments, expressions, conditionals, loops, functions, and classes. The result is stored in a context structure that feeds into ESBMC's GOTO conversion process.

## Operational models

The Python operational models — the `int`/`str`/`list`/`dict` behaviour, the
`math` and `random` models, the exception hierarchy — are **precompiled to a
GOTO binary at build time**, the way `c2goto` builds the C library. ESBMC used
to re-convert them from AST JSON, and then lower the same 113 function bodies,
on every run. A trivial verification dropped from 3.26 s to 1.80 s and the
Python regression suite from 2952 s to 1932 s
([#7747](https://github.com/esbmc/esbmc/pull/7747),
[#7778](https://github.com/esbmc/esbmc/pull/7778)).

Two further costs went with it: a module's AST is parsed on its first lookup
rather than for every `*.json` in the parser's output directory at startup
([#7777](https://github.com/esbmc/esbmc/pull/7777)), and model functions the
program cannot reach are dropped before GOTO conversion, so a program that
reaches 4 of them no longer lowers 383
([#7780](https://github.com/esbmc/esbmc/pull/7780)).

Two consequences worth knowing. Models reached through an `import` stay on the
source path, so a user module of that name can still shadow them. And `--ir`
falls back to converting from source, because the precompiled models are built
at one integer width.

## Backend: Symbolic Execution and SMT

Once the frontend produces the GOTO program, ESBMC's backend performs symbolic execution, generating instructions in Single Static Assignment (SSA) form. These are then encoded as first-order logical formulas and discharged by an SMT solver (Bitwuzla by default; Z3, MathSAT, and others are also supported).
