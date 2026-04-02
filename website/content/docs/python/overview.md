---
title: Overview
weight: 1
---

The Python frontend converts Python source files into ESBMC's internal representation (IRep), enabling the engine's bounded model checker and SMT backend to verify program properties.

> The pipeline has three stages: **AST generation** (Python) → **type annotation** (Python) → **symbol table generation** (C++). Each stage feeds into the next before ESBMC's backend takes over.

<p align="center">
  <img src="/images/python-arch.png" alt="ESBMC Python Frontend Architecture" style="max-width:65%;" />
</p>
<p align="center"><em>Python Frontend Architecture</em></p>

## AST Generation

Python code translation starts by parsing `.py` files into an Abstract Syntax Tree (AST). This is done using Python's built-in [`ast`](https://docs.python.org/3/library/ast.html) module and the [`ast2json`](https://pypi.org/project/ast2json/) package, which serializes the AST to JSON. The process runs alongside the Python interpreter and produces a JSON file for each `.py` file processed, including any imported modules.

This approach's main advantage is that it relies on a native Python module, ensuring the parsed representation faithfully reflects the language.

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

## Backend: Symbolic Execution and SMT

Once the frontend produces the GOTO program, ESBMC's backend performs symbolic execution, generating instructions in Single Static Assignment (SSA) form. These are then encoded as first-order logical formulas and discharged by an SMT solver (Boolector by default; Z3, MathSAT, and others are also supported).
