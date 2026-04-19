---
title: Creating a Frontend
---


This documentation guides the development of a new frontend for ESBMC, the Efficient SMT-Based Model Checker.

**1. No Centralized AST-to-GOTO Translation:** ESBMC does not rely on a "central" file to receive an Abstract Syntax Tree (AST) and convert it to a GOTO program. This is because the GOTO intermediate representation does not follow a fixed grammar; GOTO is generated during compilation. The translation is tightly integrated with the processing steps. You can see below an example of an AST node (`x:int = 10`) using the JSON format:

```json
{
    "_type": "AnnAssign",
    "annotation": {
        "_type": "Name",
        "col_offset": 2,
        "ctx": {
            "_type": "Load"
        },
        "end_col_offset": 5,
        "end_lineno": 1,
        "id": "int",
        "lineno": 1
    },
    "target": {
        "_type": "Name",
        "col_offset": 0,
        "ctx": {
            "_type": "Store"
        },
        "end_col_offset": 1,
        "end_lineno": 1,
        "id": "x",
        "lineno": 1
    },
    "value": {
        "_type": "Constant",
        "col_offset": 8,
        "end_col_offset": 10,
        "end_lineno": 1,
        "kind": null,
        "lineno": 1,
        "n": 10,
        "s": 10,
        "value": 10
    }
}
```

You can find below the GOTO representation of this AST (using the command `esbmc main.py --goto-functions-only`):

````C
__ESBMC_main (__ESBMC_main):
        // 17 file assign.py line 1 column 0
        ASSIGN x=10;
        // 18 no location
        END_FUNCTION
````

Note that the Jimple frontend has an [entry point](https://github.com/esbmc/esbmc/blob/master/src/jimple-frontend/jimple-converter.cpp) to convert to GOTO, which can serve as an example.

Here is an example of how to convert an AST from a file (in the case of Jimple there is only one class per file): [example](https://github.com/esbmc/esbmc/blob/master/src/jimple-frontend/AST/jimple_file.cpp#L108-L170).

Lastly, a main function also needs to be added: https://github.com/esbmc/esbmc/blob/master/src/jimple-frontend/jimple-language.cpp#L125-L205.

**2. Main Interface – languaget:** The key interface to handle source file processing and GOTO program generation is the [languaget](https://github.com/esbmc/esbmc/blob/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/util/language.h#L15) class. Every frontend in ESBMC, including C/C++, Python, and Jimple, implements this interface. When creating a new frontend, you'll need to develop your implementation of this interface to handle the input language and generate the corresponding GOTO program.

**3. Using ASTs from Other Frontends:** If your new frontend involves generating an AST (Abstract Syntax Tree), reviewing the existing implementations of the [Python](https://github.com/esbmc/esbmc/tree/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/python-frontend) and [Jimple](https://github.com/esbmc/esbmc/tree/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/jimple-frontend) frontends might be helpful. These provide examples of translating AST nodes to GOTO instructions within ESBMC. For reference, see the [Jimple Frontend Pull Request](https://github.com/esbmc/esbmc/pull/503).

**4. Required Implementation Files:** In the case of integrating a frontend for a new programming language, you'll need to follow the file structure and functionality of the existing C/C++ frontend. Specifically, you should implement the following files:

* [_clang_c_language:_](https://github.com/esbmc/esbmc/blob/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/clang-c-frontend/clang_c_language.h) Handles language-specific operations and the integration with ESBMC's core analysis engine.
* [_clang_c_convert:_](https://github.com/esbmc/esbmc/blob/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/clang-c-frontend/clang_c_convert.h) Converts AST nodes into GOTO instructions.
* [_clang_c_adjust:_](https://github.com/esbmc/esbmc/blob/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/clang-c-frontend/clang_c_adjust.h) Performs final adjustments on the GOTO program, ensuring it conforms to the desired execution model and constraints.

**5. Additional Considerations**

* _Symbol Table:_ In ESBMC, a program consists of a symbol table and functions. This information is stored inside a [`contextt` class](https://github.com/esbmc/esbmc/blob/master/src/util/context.h#L26).

* _Build System:_ Ensure your new frontend is included in the CMake build system. You must modify [CMakeLists.txt](https://github.com/esbmc/esbmc/blob/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/CMakeLists.txt) to register the new language and its dependencies.

* [_SMT Solvers:_](https://github.com/esbmc/esbmc/tree/82b3069a567dafd299f7f0d3916388a0c5cedb06/src/solvers) ESBMC supports various state-of-the-art SMT solvers (e.g., Z3, Boolector), so ensure your frontend generates GOTO programs compatible with these backends.
* [_Testing:_](https://github.com/esbmc/esbmc/tree/82b3069a567dafd299f7f0d3916388a0c5cedb06/regression) Once implemented, extensive testing is required. You can leverage ESBMC’s existing regression test suite to check the correctness of the GOTO generation and validation processes.

**6. Development Resources**

* Explore ESBMC’s [source code repository](https://github.com/esbmc/esbmc) for further insights into existing frontends.

* The community discussions and open pull requests on GitHub often contain valuable information about ongoing development efforts related to new language support.

* Following these steps, you can successfully integrate a new language frontend into ESBMC.