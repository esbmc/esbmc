# A JavaScript frontend for ESBMC

This document proposes a JavaScript frontend built as a *sibling* of the
existing Python frontend rather than as a standalone compiler. The guiding
constraint is maximum reuse: every mechanism the Python frontend already uses to
get a dynamically typed language into `irep2` is assessed for direct reuse,
generalisation, or replacement, and anything new has to justify itself against
the alternative of extending what is there.

The short version of the conclusion:

- **The pipeline shape is reusable verbatim.** External parser → JSON AST →
  annotation pass → C++ converter → legacy `exprt`/`codet` into `contextt` →
  `clang_cpp_adjust` → `goto_convert`. Nothing about that is Python-specific.
- **The runtime substrate is reusable after generalisation.** `PyObject`,
  `PyListObject`, the string handler, the dict handler, the exception lowering
  and the class-to-`struct_typet` builder all become a shared
  *dynamic-language layer* that Python and JavaScript both consume.
- **The typing strategy is not reusable as-is, and this is the crux of the
  proposal.** ESBMC's Python frontend is, in practice, a static
  monomorphising translator that leans on PEP 484 annotations; its genuinely
  dynamic path is narrow enough that assigning a non-literal to a
  dynamically-typed variable raises "not yet supported"
  (`src/python-frontend/converter/converter_stmt.cpp:3243`). JavaScript has no
  annotations. The frontend therefore needs a real inference pass and a
  first-class tagged-value path — and building that is the one large piece of
  new infrastructure, which pays back into Python immediately.

Everything below is evidence for those three claims and a plan that follows from
them.

---

## 1. What already exists

### 1.1 irep2

`irep2` (`src/irep2/README.md`) is a typed, refcounted, copy-on-write term
representation. Two X-macro manifests — `expr_kinds.inc` and `type_kinds.inc` —
are the single source of truth for every node kind; adding a kind is a manifest
row plus a class with a `fields` tuple, and comparison, CRC, printing and
operand iteration are generated.

The pieces that matter for a dynamic language:

| Need | irep2 construct | Status |
|---|---|---|
| Machine integers | `signedbv` / `unsignedbv` | present |
| IEEE doubles | `floatbv`, `ieee_add`/`ieee_mul`/…, `isnan`/`isinf` | present |
| Records | `struct` (members, names, pretty names, tag) | present |
| Tagged unions | `union` + `constant_union` | present |
| Heap references | `pointer`, `address_of`, `dereference`, `dynamic_object` | present |
| Indexable storage | `array` (fixed, dynamic, **infinite**) + `index` | present |
| Functions and indirect calls | `code` type, `code_function_call`, function pointers | present |
| Exceptions | `code_cpp_throw` / `code_cpp_catch` | present |
| Field read/update | `member`, `with`, `byte_extract`/`byte_update` | present |
| Runtime type predicates | `isinstance`, `hasattr`, `isnone` | **present, and already dynamic-language-specific** |

That last row is the important one. `expr_kinds.inc` already carries three
"Python-specific predicates" (`irep2/README.md`, *Python-specific predicates*),
and `isinstance2t` is resolved during symbolic execution by querying the
points-to set
(`src/goto-symex/builtin_functions/python_builtins.cpp:19-120`). These are not
Python predicates in any deep sense — they are *dynamic-language* predicates, and
JavaScript's `typeof`, `instanceof` and `in` map onto exactly this machinery.
Renaming them is optional; reusing them is not in doubt.

**Verdict: no new irep2 node kinds are required for the MVP.** The `README.md`
checklist for adding a kind exists if a later milestone needs one (a dedicated
`getprop` node is discussed in §4.4), but the MVP maps entirely onto existing
terms.

### 1.2 The Python frontend, end to end

```
  main.py
    │  python_languaget::parse()                       python_language.cpp:92
    │    spawns `python3 parser/__main__.py`  ← FLAIL-mangled .py files, dumped
    │    to a temp dir at first use (dump_python_script(), :63)
    ▼
  <module>.json                     ast + ast2json, vendored in libs/ast2json
    │  preprocessor/*.py (22 mixins, ~11k lines) desugars in Python
    │  python_annotation<json>      adds type nodes to the JSON  (:243)
    ▼
  annotated JSON AST
    │  python_converter::convert()                     python_converter.cpp
    │    ~65 translation units: converter/, python-list/, python-dict/,
    │    string/, set/, tuple/, class/, exception/, math/, numpy/, lambda/
    ▼
  contextt  (legacy symbolt / exprt / codet)
    │  add_cprover_library()   ← links the C operational models  (:286)
    │  clang_cpp_adjust        ← the C++ frontend's adjuster, reused wholesale
    ▼
  goto_convert → goto-symex → SMT
```

Registration is three edits: `language_idt::PYTHON` and the `.py` extension in
`src/langapi/mode.{h,cpp}`, and `LANGAPI_MODE_PYTHON` in the `mode_table` at
`src/esbmc/globals.cpp:15`.

### 1.3 How Python values are actually represented

This is where the reuse analysis has to be precise, because the headline
"ESBMC supports a dynamically typed language" is doing more work than the
implementation supports.

**Scalars are statically monomorphised.** `type_handler::get_typet`
(`src/python-frontend/type/type_handler.cpp:429`) maps annotation strings to
concrete types:

| Python | ESBMC type |
|---|---|
| `int` | `signedbv(64)` — with a `FIXME: Support bignum` |
| `float` | `floatbv(52,11)` (C `double`) |
| `bool` | `bool` |
| `str` | `char[N]` / `char *` |
| `None` | `pointer(bool)` (`none_type()`, `src/util/lang/python_types.cpp:4`) |
| `Any` | `pointer(empty)` — i.e. `void *` (`:10`) |
| `Callable` | `pointer(code)` |
| class `C` | `struct_typet` tagged `C`, allocated via `__ESBMC_new_object` |

So a Python `int` variable is a 64-bit machine integer for the whole program.
There is no per-assignment tag.

**Tags exist, but mostly inside containers.** The tagged representation is

```c
typedef struct __ESBMC_PyObj {
  const void *value;    /* pointer to the payload            */
  size_t float_idx;     /* index into a real-sorted side buffer, for --ir */
  size_t type_id;       /* hash of the type name             */
  size_t size;          /* payload byte width                */
} PyObject;                       /* src/c2goto/library/python/python_types.h:33 */
```

`type_id` is `std::hash<std::string>{}(type_name)` truncated to the address
width (`type_handler.cpp:1137`). Lists are

```c
typedef struct __ESBMC_PyListObj {
  PyType *type; PyObject *items; size_t size;
} PyListObject;                                              /* :53 */
```

with `items` an *infinite* array produced by `__ESBMC_create_inf_obj()` and
materialised by symex — the SMT-level unbounded-array trick that makes symbolic
list indices tractable.

**The dynamic-variable path is deliberately narrow.** A scalar variable only
becomes `PyObject`-typed when `scalar_tag_candidates()` flags it at an `if`
join point, and even then:

```cpp
if (is_tagged) {
  if (ast_node["value"]["_type"] == "Constant") { get_tagged_scalar_assign(...); return; }
  throw std::runtime_error(
    "assigning a non-literal value to a dynamically-typed variable is "
    "not yet supported");
}
```
— `src/python-frontend/converter/converter_stmt.cpp:3224-3245`

That is the honest state of the art: tagged scalars support literal assignment
at branch joins, and nothing else.

**Classes reuse the C++ object model.** `python_class_builder::build`
(`src/python-frontend/class/python_class_builder.cpp`) emits a `struct_typet`
with a method table and base classes, instances are allocated through
`__ESBMC_new_object` (`src/c2goto/library/python/list.c:14`, intercepted by
symex's `symex_mem_inf` to produce a non-expiring typed object), and
`clang_cpp_adjust` — the C++ frontend's adjuster — runs over the result
(`python_language.cpp:314`).

**Exceptions reuse the C++ exception model.** `raise` becomes a `cpp-throw`
side effect and `try`/`except` a `cpp-catch`
(`src/python-frontend/exception/python_exception_handler.h`), lowered by
`goto_convert` and `src/goto-programs/remove_exceptions.cpp`, with type ids
shared with `PyObject.type_id` (`src/goto-programs/exception_globals.h:29`).

**Two model layers.** C operational models
(`src/c2goto/library/python/{list,string,math,scalar,slice}.c`, ~4.2k lines)
compiled to GOTO by `c2goto` and linked by `add_cprover_library`; and
*Python-level* models (`src/python-frontend/models/*.py`, 28 modules) which are
FLAIL-mangled into the binary and parsed by the frontend itself. The second
layer is the cheap one — `math`, `random`, `os`, `re`, `collections` and the
exception hierarchy are all just Python source.

### 1.4 Where the Python frontend is weak

Stated plainly, because the JS design has to plan around it:

1. **No general dynamic scalar.** As above.
2. **Iteration is special-cased.** "Only `for` loops using the `range()`
   function are supported" (`src/python-frontend/README.md`, *Limitations*); the
   preprocessor rewrites other shapes where it can.
3. **Closures do not capture.** Lambdas become standalone function symbols with
   parameters assumed `double`, stored as function pointers
   (`src/python-frontend/lambda/python_lambda.h`); "higher-order and nested
   lambda expressions are not supported".
4. **`try`/`finally` is restricted.** `body_has_escaping_control_flow` refuses
   shapes where a `return`/`break`/`continue` would skip the appended `finally`.
5. **Ints are 64-bit.** Bignum is a `FIXME`.

JavaScript needs (1), (2), (3) and (4) to be *good*, not special-cased. That
determines the roadmap ordering in §10.

---

## 2. Parser selection

### 2.1 Candidates

| | AST | Deps | ES coverage | Embedding | Verdict |
|---|---|---|---|---|---|
| **Acorn** 8.18.0 | ESTree | **none** | stage-4 only, `ecmaVersion: "latest"` | pure JS, one file, MIT | **chosen** |
| Babel parser 8.0.4 | ESTree superset | several `@babel/*` | + proposals, JSX, Flow, TS | pure JS, heavier | fallback for TS/JSX |
| SWC | custom Rust AST | Rust toolchain | full + TS | Rust cdylib or napi | rejected |
| Tree-sitter | CST, error-tolerant | C lib + grammar | grammar-defined | C library — easiest to link | rejected |
| TypeScript Compiler API | TS AST | full `typescript` package | full + type checker | Node only, ~60 MB | deferred (see §2.4) |

### 2.2 Why Acorn

- **ESTree is the lingua franca.** Acorn defines the de-facto ESTree output that
  espree/ESLint and most tooling consume, so every downstream tool, test corpus
  and reference implementation speaks the same node shapes. A JSON dump of an
  ESTree tree is directly analogous to what `ast2json` produces for Python — the
  converter reads `node.type` the way `python_converter` reads `_type`.
- **Zero runtime dependencies, MIT, one file.** This matters because ESBMC vendors
  its parser support. Acorn drops into `src/js-frontend/libs/acorn/` the same
  way `ast2json` is vendored at `src/python-frontend/libs/ast2json/`.
- **Stage-4 only is a feature, not a limitation.** ESBMC needs to model
  semantics it can defend. A parser that accepts proposals ESBMC cannot lower
  produces a converter that fails late and confusingly.
- **`acorn-walk` and `acorn-loose` are in the same repo**, giving a supported
  traversal utility for the annotation pass and an error-tolerant mode for
  diagnostics.

### 2.3 Why not the others

- **SWC.** The performance argument does not survive the interface. Published
  benchmarking of Rust JS parsers consistently finds that serialising the Rust
  AST across the boundary eats most of the parse-time win — and ESBMC *must*
  cross that boundary, because the converter is C++. Against that we would take
  on a Rust toolchain in ESBMC's build, a non-ESTree AST, and a second
  dependency-management story. Parse time is not ESBMC's bottleneck: the Python
  frontend spends ~0.6–1.2 s in "GOTO program creation" and milliseconds in the
  solver on small programs (`src/python-frontend/README.md`, Examples 2 and 5).
  Optimising the cheapest stage is the wrong trade.
- **Babel parser.** Strictly more capable and strictly heavier: a dependency
  tree instead of one file, and an AST that is an ESTree *superset*, so the
  converter would have to handle node shapes that only appear under
  proposal plugins. Worth revisiting only when JSX or TypeScript syntax becomes
  a goal, at which point it is a drop-in for the same JSON contract.
- **Tree-sitter.** The easiest to *link* (it is a C library, so no subprocess at
  all) and the worst fit for the *task*. It produces a concrete syntax tree
  optimised for incremental editor reparsing, not a semantic AST: no scope
  resolution, no early-error checking, punctuation nodes throughout, and no
  ESTree compatibility. The converter would spend its budget rediscovering
  structure Acorn hands over for free. The error tolerance that makes it great
  for editors is irrelevant to a verifier that should reject malformed input.
- **TypeScript Compiler API.** The only candidate that brings a *type checker*,
  which is genuinely attractive given §1.4's typing problem — `tsc` could supply
  the inference the annotation pass otherwise has to do itself. But it requires
  Node, is two orders of magnitude larger than Acorn, and its AST is
  TypeScript-shaped. Deferred deliberately, not dismissed: see §2.4.

### 2.4 Embedding: subprocess first, then in-process

**Stage 1 (MVP) — Node subprocess, mirroring Python exactly.**
`js_languaget::parse()` FLAIL-dumps `acorn.js` plus a small `parse.js` driver to
a temp directory and runs `node parse.js <file> <outdir>`, which writes
`<module>.json`. This is a direct structural copy of
`python_language.cpp:92-262`, including the interpreter-discovery and
version-check logic, and it lets milestone 1 land in days rather than weeks.

The cost is inheriting the Python frontend's most-reported operational
complaint — an external interpreter on `PATH` (the project's own build
documentation carries an "Important: the Python frontend needs `python3` on
`PATH`" warning).

**Stage 2 — embed QuickJS-ng and drop the Node dependency.**
Acorn is dependency-free ECMAScript, so it does not need Node; it needs *a* JS
engine. QuickJS-ng is a few C files with no external dependencies, MIT
licensed, and supports well beyond the ES2020 subset Acorn's own source uses.
Statically linking it and running the FLAIL-mangled Acorn inside the ESBMC
process removes the subprocess, the `PATH` lookup, the temp-directory dance and
the JSON round-trip through the filesystem — the converter can read the parse
result straight out of the engine.

This is a strictly better end state than Python's, and it is the reason to
choose a *pure-JS* parser over a native one: pure JS can be embedded in a 400 KB
engine, whereas SWC or Tree-sitter would each pull their own native toolchain.
Stage 2 is scheduled as milestone 7 so it is never on the critical path.

**Contract between stages.** Both stages produce byte-identical JSON, so the
converter is unaware of which is in use, and Stage 1 remains a supported
fallback (`--js-parser=node|embedded`).

---

## 3. Construct-by-construct reuse analysis

`R` = reuse as-is · `G` = generalise the Python component · `N` = new.

| JavaScript | Python analogue | Representation | Runtime model | irep2 mapping | Verdict |
|---|---|---|---|---|---|
| `number` | `float` | `floatbv(52,11)` | — | `ieee_*`, `isnan` | **R** |
| `bigint` | `int` (bignum FIXME) | wide `signedbv` | — | `add`/`mul` | **G** (shared gap) |
| `string` | `str` | `char[N]` / `char *` | `string.c`, `string_handler` | `index`, `constant_string` | **R** (UTF-16 caveat) |
| `boolean` | `bool` | `bool` | — | `constant_bool` | **R** |
| `null` | `None` | `pointer(bool)` sentinel | — | `isnone2t` | **R** |
| `undefined` | *(none)* | second sentinel | — | `isnone2t` variant | **N** (small) |
| `symbol` | *(none)* | opaque unique id | — | `unsignedbv` | **N** (deferred) |
| array | `list` | `JsArrayObject` | `list.c` | infinite `array` + `index` | **G** |
| object (fixed shape) | class instance | `struct_typet` | `__ESBMC_new_object` | `member`, `with` | **R** |
| object (dynamic keys) | `dict` | `DynDict` | `python-dict/` | struct + arrays | **G** |
| class | class | `struct_typet` + methods | `python_class_builder` | `member`, indirect call | **R** |
| inheritance | inheritance | base subobject | `get_bases`, `clang_cpp_adjust` | `member` chain | **R** |
| prototype chain | *(none)* | `__proto__` link | new OM | convert-time walk / OM loop | **N** |
| function | function | function symbol | — | `code_function_call` | **R** |
| closure | nested function | `{fn, env}` pair | `__ESBMC_new_object` env | fn-pointer call | **N** (see §4.3) |
| `this` | `self` | leading parameter | — | — | **R** |
| exception | exception | `cpp-throw`/`cpp-catch` | `remove_exceptions` | `code_cpp_throw` | **R** |
| `finally` | `finally` (restricted) | — | — | — | **G** |
| dynamic typing | annotations + narrow tags | `DynObject` | shared tag OM | `isinstance2t` | **G** — the big one |
| `for...of` | `for … in range()` | desugared | — | `code_goto` loop | **G** |
| modules (ESM) | `import` | module manager | `module_manager` | — | **G** |
| `typeof` | `type(x)` | tag read | — | `isinstance2t` | **R** |
| `instanceof` | `isinstance` | value-set query | `python_builtins.cpp` | `isinstance2t` | **R** |
| `in` | `in` (dict) | key membership | `python-dict/` | `hasattr2t` | **R** |

Aggregate: of 24 constructs, 11 reuse directly, 7 generalise a Python
component, 6 are new — and of those six, three (`undefined`, `symbol`,
prototype chains) are small and two (closures, dynamic typing) are the load
bearing work.

---

## 4. Mapping JavaScript into irep2

### 4.1 Primitives

**`number`.** One type, IEEE-754 binary64: `floatbv(52,11)`, the same
`double_type()` the Python frontend produces for `float`. Arithmetic maps to
`ieee_add`/`ieee_sub`/`ieee_mul`/`ieee_div` with the rounding-mode operand, and
`Number.isNaN`/`isFinite` map to `isnan2t`/`isfinite2t`.

Two JavaScript-specific obligations:

- **Bitwise operators are `ToInt32`.** `a | 0`, `a << 1`, `a & 0xff` and `~a`
  all convert through a modular 32-bit signed integer. Lower as
  `typecast(floatbv → signedbv(32))` under JS's truncate-toward-zero-modulo-2³²
  rule, apply the `signedbv(32)` bit operation, and cast back. `>>>` uses
  `unsignedbv(32)`. This is a small, self-contained conversion helper — the
  single most common source of unsoundness in naive JS models, so it belongs in
  milestone 4, not later.
- **Integer-valued fast path.** Most computational JS uses numbers that are
  integers. The annotation pass (§5) marks a variable *integral* when every
  reaching definition is an integer literal, an integer-preserving operation, or
  a `| 0`; such variables are emitted as `signedbv(64)` with an overflow guard
  rather than `floatbv`. This is exactly the widening the Python annotator does
  for `Union`/`Any`, run in the opposite direction, and it is where JS
  verification performance will come from — bit-blasting doubles is far more
  expensive than bit-blasting 64-bit integers.

**`bigint`.** irep2's `constant_int` is already arbitrary-precision (`BigInt`),
but every *type* is a fixed-width bitvector, so an exact model needs either a
bignum type kind or an operational model over a digit array. The MVP takes the
same approximation Python takes — a configurable-width `signedbv`
(`--js-bigint-width`, default 128) with an overflow property so the
approximation is *reported* rather than silent. Note this is a **shared** gap:
solving it properly fixes Python's `FIXME: Support bignum` at the same time,
which is the argument for doing it in the shared layer.

**`string`.** Reuse the Python model directly: `char[N]` for literals, `char *`
for computed strings, with the ~9.5k lines of `string/` plus
`src/c2goto/library/python/string.c` behind it. The
overlap between Python's and JavaScript's string methods is large
(`indexOf`/`find`, `slice`, `split`, `replace`, `startsWith`, `trim`/`strip`,
`padStart`/`rjust`, `toUpperCase`/`upper`, …), so most JS methods are a name
mapping onto an existing implementation.

*Known deviation:* JavaScript strings are sequences of UTF-16 code units;
Python's model here is byte/char oriented. The MVP restricts sound reasoning to
the ASCII/Latin-1 range and emits a diagnostic on non-Latin-1 literals. Full
UTF-16 (surrogate pairs, `codePointAt`, `length` counting units not code points)
is a milestone-9 item and again benefits both frontends.

**`boolean`.** `bool_type()`. JS truthiness (`""`, `0`, `NaN`, `null`,
`undefined` falsy) becomes an explicit `ToBoolean` lowering — Python already has
the analogous truthiness lowering for `any()` and `if` conditions
(`src/python-frontend/function_call/builtins.cpp`), so this is a table extension.

**`null` and `undefined`.** JavaScript distinguishes them, Python has only
`None`. Represent both as distinct singleton sentinels:

```cpp
typet js_null_type()      { return pointer_typet(bool_typet()); }   // == none_type()
typet js_undefined_type() { return pointer_typet(char_typet()); }   // distinct
```

with distinct `type_id`s in the shared tag namespace, so `x === null` and
`x === undefined` are separate predicates while `x == null` (loose) is their
disjunction. On the monomorphic path, a variable that may be `undefined` uses
the optional-type builder Python already has
(`type_handler::build_optional_type`, `type_handler.cpp:1436`).

**`symbol`.** Deferred past the MVP. When needed: an opaque `unsignedbv(64)`
identity allocated by a counter, with `Symbol()` calls yielding fresh distinct
values, and well-known symbols as reserved constants.

### 4.2 Objects and arrays

The single most important modelling decision, because it decides whether
verification is fast or hopeless.

**Arrays.** Generalise `PyListObject` to

```c
typedef struct __ESBMC_JsArrayObj {
  DynType   *type;
  DynObject *items;   /* infinite array, materialised by symex */
  size_t     length;
} JsArrayObject;
```

which is `PyListObject` with the field renamed. The infinite-array trick
(`__ESBMC_create_inf_obj`) carries over unchanged and is precisely what makes
symbolic indices tractable. JS-specific behaviour that Python's list lacks:
out-of-bounds read yields `undefined` rather than raising (so the bounds
*property* becomes a warning-level check, configurable via
`--js-array-oob=undefined|error`); assigning past the end extends `length`;
holes are `undefined`. Each is a modification to the array OM, not a new model.

**Objects — two tiers, chosen per allocation site.**

*Tier 1: shape-inferred objects → `struct_typet`.* When the annotation pass can
prove an object's key set is fixed (an object literal whose properties are never
added to or deleted, which covers the overwhelming majority of computational
JS), emit a `struct_typet` exactly as `python_class_builder` does for a class
instance. Then `o.x` is `member2t`, `o.x = v` is `with2t`/`code_assign`, and the
whole thing costs the solver what a C struct costs. Instances are allocated by
`__ESBMC_new_object` so they get reference semantics and survive escaping their
defining scope — the mechanism Python class instances already use.

*Tier 2: open objects → `DynDict`.* When keys are computed (`o[k]`), added, or
deleted, fall back to the generalised dict handler (`src/python-frontend/
python-dict/`, ~4k lines across 8 translation units) keyed by string with
`DynObject` values. Slower, general, already written.

The tier is a per-allocation-site decision, so one open object does not degrade
the rest of the program.

**Classes.** ES6 `class` maps onto `python_class_builder` with almost no
change: fields become struct members, methods become symbols registered in the
method table, `extends` becomes a base subobject, `super()` becomes a base
constructor call, and `clang_cpp_adjust` resolves the lot. `constructor` is
`__init__`; `static` members are class-level attributes, which the Python builder
already handles. `get`/`set` accessors are new: lower property access on an
accessor-bearing shape to the corresponding method call at conversion time.

**Modules.** ESM `import`/`export` maps onto `module_manager` /
`module_locator` / `global_scope` (`src/python-frontend/module/`) — the same
problem of resolving a name to a file, parsing it, and merging its symbols. The
JS-specific parts are extension resolution, `package.json` `"exports"`, and
default-vs-named exports; live bindings are the one genuine semantic difference
and are handled by treating an imported binding as a read through the exporting
module's symbol rather than a copy. CommonJS `require` is deferred (Acorn's
`sourceType: "commonjs"` parses it; resolving it needs the Node resolution
algorithm).

### 4.3 Closures

Python's lambda support does not capture, so this is new work — and it is
unavoidable, because closures are not an advanced JavaScript feature, they are
how JavaScript is written.

**Closure conversion in the JS-side preprocessor**, mirroring how
`src/python-frontend/preprocessor/*.py` desugars before the C++ converter ever
runs:

1. Compute free variables per function (Acorn gives complete scope structure;
   `acorn-walk` supplies the traversal).
2. For each function with free variables, synthesise an environment struct
   holding one field per captured binding — by *reference* if the binding is
   ever reassigned after capture (JS captures variables, not values), by value
   otherwise.
3. Lambda-lift the function to top level with the environment as a leading
   parameter (structurally identical to how methods take `self`/`this`).
4. Represent a closure value as `struct JsClosure { code *fn; void *env; }`,
   with the environment allocated by `__ESBMC_new_object` so it outlives the
   creating frame.
5. A call `f(a)` where `f` is a closure becomes
   `code_function_call(ret, dereference(f.fn), {f.env, a})`.

irep2 needs nothing new: `code` type, `pointer(code)`, `address_of`,
`dereference`, `code_function_call`. Symex resolves the indirect call through
the value set (`src/goto-symex/symex_function.cpp`).

**Known limitation, stated up front:** k-induction refuses function-pointer
calls (`symex_function.cpp:778`, "k-induction does not support function pointer
calls yet"). Closure-heavy code is therefore BMC-only until that is addressed.
Where the annotation pass can prove a callee is unique — the common case for a
`const f = () => …` called directly — it devirtualises to a direct call at
conversion time, which sidesteps the restriction entirely and is also much
cheaper to solve.

### 4.4 Property lookup and prototype chains

Property access is the hottest operation in JavaScript and needs to resolve
statically wherever possible.

**Resolution ladder**, cheapest first:

1. **Known shape, own property** → `member2t`. Free.
2. **Known shape, inherited property** → walk the `__proto__` chain *at
   conversion time* and emit `member2t` on the resolved subobject. This is
   exactly what C++ does for base-class members, and `clang_cpp_adjust` already
   implements the subobject arithmetic.
3. **Known shape, accessor** → emit the getter/setter call.
4. **Open object, constant key** → dict lookup with a constant key.
5. **Open object, computed key** → dict lookup with a symbolic key; if the
   receiver's shape set is finite and known, emit an `if2t` chain over the
   candidate shapes instead, which keeps the fast path for the common
   "two possible shapes" case.
6. **Fully unknown** → the runtime OM `__ESBMC_js_getprop(obj, key)`, a loop over
   the prototype chain bounded by `--js-proto-depth` (default 8), with an
   unwinding assertion so exceeding the bound is *reported*, never silently
   truncated.

Steps 1–4 cover essentially all MVP-scope programs. Step 6 exists so that
nothing is silently wrong.

**Alternative considered and rejected for the MVP:** a dedicated `getprop2t`
irep2 node resolved during symex the way `isinstance2t` is. It is the right
long-term answer — symex knows the points-to set and could resolve shapes that
the converter cannot — but it commits to a new node kind and a symex lowering
before there is evidence the convert-time ladder is insufficient. Revisit at
milestone 9 with benchmark data.

### 4.5 Exceptions

Direct reuse. `throw e` becomes a `cpp-throw` side effect, `try`/`catch` becomes
`cpp-catch`, and `goto_convert` + `src/goto-programs/remove_exceptions.cpp`
lower it. Two JS-specific pieces:

- **JS throws any value**, not just objects. Non-object throws are boxed in a
  synthetic `__ESBMC_JsThrown` struct so the `exception_typeid` machinery has a
  type to dispatch on. A bare `catch (e)` catches everything, which is the
  degenerate and easy case.
- **`finally` must work properly.** Python's handler refuses shapes where
  `return`/`break`/`continue` escapes the `try`
  (`python_exception_handler.h`, `body_has_escaping_control_flow`). JavaScript
  uses `try`/`finally` for resource cleanup constantly, so this needs a real
  lowering: duplicate the `finally` block onto every exit edge (normal
  completion, each `return`, each `break`/`continue` crossing the boundary, and
  the exceptional path), which is the standard treatment and is a
  `goto_convert`-level transformation. Doing it in the shared layer lifts
  Python's restriction too.

The built-in error hierarchy (`Error`, `TypeError`, `RangeError`,
`ReferenceError`, `SyntaxError`) is declared in a *JavaScript* model file, the
way `src/python-frontend/models/exceptions.py` declares Python's — no C++
required.

---

## 5. Dynamic typing strategy

### 5.1 The honest starting position

Python's approach is *annotation-directed static monomorphisation with a narrow
tagged escape hatch*. JavaScript has no annotations, so the annotation-directed
half is missing and the escape hatch is too narrow to carry the load (§1.3:
non-literal assignment to a tagged variable throws).

Two options:

- **(a)** Tag everything: every value is a `DynObject`, every operation is a
  runtime-dispatched OM call.
- **(b)** Infer aggressively, tag only where inference fails.

**(a) is correct and unusably slow.** Every `a + b` becomes a tag test plus a
branch over string-concat / numeric-add / `valueOf` coercion, every variable is
a heap cell, and the SMT formula for a loop that ESBMC currently solves in
milliseconds acquires a dispatch tree per iteration. It also throws away the
integer fast path (§4.1) that makes numeric JS tractable at all.

**(b) is what the Python frontend already does**, minus the inference. So the
proposal is (b): keep the architecture, replace the annotation source with
inference, and *widen the tagged path* from "literals at `if` joins" to a real
one.

### 5.2 The inference pass

A `js_annotation` pass, structurally parallel to `python_annotation<json>`
(`src/python-frontend/python_annotation/`), running over the ESTree JSON before
the C++ converter. Because Acorn resolves scopes, this is a standard
flow-insensitive-with-SSA-refinement analysis:

1. Build the scope tree and bind every identifier to a declaration
   (`var` hoisting, `let`/`const` TDZ, function declarations, parameters,
   destructuring).
2. Assign each binding a **type lattice** element:
   `⊥ < {undefined, null, boolean, integral-number, number, bigint, string,
   symbol, shape(S), array(T), closure(F), object} < ⊤`.
3. Propagate to a fixpoint over reaching definitions, joining at merge points
   and widening at loop back-edges.
4. Annotate every AST node with its inferred type — the same JSON-node-annotation
   contract `python_annotation` uses, so the converter's `get_typet` reads
   an annotation whether it came from a PEP 484 hint or from inference.
5. Refine with local **type narrowing**: `typeof x === "string"`,
   `Array.isArray(x)`, `x instanceof C`, `x === null`, truthiness guards. These
   are the idioms real JS uses to be safe, and honouring them is what keeps most
   variables monomorphic in practice.

Where step 3 lands on a singleton, the converter emits a concrete type. Where it
lands on a union or `⊤`, the converter emits a tagged `DynObject`.

### 5.3 The tagged path, widened

The `DynObject` layout is `PyObject` renamed, with the same
`{value, float_idx, type_id, size}` shape and the same `type_id` allocation, so
tags are shared across languages. What must be *built* — and what the Python
frontend today refuses — is:

- `dyn_assign(lvalue, rvalue)` for an arbitrary rvalue, not just a literal:
  evaluate the rvalue, box it into the tag struct (allocating payload storage
  through the existing `__ESBMC_copy_value` path, including its `--ir` float
  side-buffer handling), and assign the struct.
- `dyn_binop(op, a, b)` dispatching on `(a.type_id, b.type_id)` — where the
  inference pass has narrowed the possible tag set, emit only those arms, so a
  variable known to be "string or number" costs a two-way branch rather than a
  full dispatch table.
- `dyn_unbox(v, T)` with a *checked* unbox that raises `TypeError` on tag
  mismatch, so type confusion is a reported property rather than silent
  reinterpretation.
- `dyn_truthy(v)`, `dyn_typeof(v)`, `dyn_equals(v, w)` for `==` vs `===`.

`typeof` is a `type_id` read; `instanceof` is `isinstance2t`, already resolved by
symex against the points-to set; `in` is `hasattr2t`. So the *predicates* are
free — it is boxing, dispatch and coercion that are new.

### 5.4 SMT encoding implications

- A monomorphic `number` costs one `floatbv`; an integral-inferred number costs
  one `signedbv(64)`, which is materially cheaper to bit-blast.
- A tagged value costs a 4-field struct plus a pointer dereference per read, and
  each dispatch adds one `if2t` arm per candidate tag. Narrowing the candidate
  set is therefore the whole game — hence the emphasis on step 5 above.
- Shape-inferred objects cost exactly what C structs cost; open objects cost
  what the dict OM costs.
- The `--ir` (integer/real) mode's float side-buffer
  (`__ESBMC_float_buf`, `src/c2goto/library/python/list.c:27`) is a
  tag-layer concern and carries over unchanged.

### 5.5 The shared dynamic-language layer

**Yes — Python and JavaScript should share a runtime layer, and this is the
core architectural recommendation.**

```
                     ┌───────────────┐  ┌───────────────┐
   .py ──▶ ast2json ─▶│ python_       │  │ js_converter  │◀─ acorn ◀── .js
                     │ converter     │  │               │
                     └───────┬───────┘  └───────┬───────┘
                             │                  │
                     ┌───────▼──────────────────▼───────┐
                     │      src/dynlang/  (new)         │
                     │  ─ dyn_value: tag protocol,      │
                     │    type_id allocation, box/unbox │
                     │  ─ dyn_container: array + dict   │
                     │    handlers (from python-list/,  │
                     │    python-dict/)                 │
                     │  ─ dyn_string: string_handler    │
                     │  ─ dyn_object: struct/shape      │
                     │    builder (from class/)         │
                     │  ─ dyn_exception: throw/catch/   │
                     │    finally lowering              │
                     │  ─ dyn_scope: symbol_id, module  │
                     │    manager, closure conversion   │
                     └───────────────┬──────────────────┘
                                     │
                     ┌───────────────▼──────────────────┐
                     │ src/c2goto/library/dyn/*.c       │
                     │  DynObject, DynArray, DynDict,   │
                     │  string ops   (from python/*.c)  │
                     └───────────────┬──────────────────┘
                                     │
                     contextt ▶ clang_cpp_adjust ▶ goto_convert
                              ▶ goto-symex ▶ SMT
```

The layer is created by **moving** Python code, not copying it:
`src/python-frontend/{python-list,python-dict,string,set,tuple,class,exception}/`
and `src/c2goto/library/python/*.c` relocate to `src/dynlang/` and
`src/c2goto/library/dyn/`, with `PyObject`/`PyListObject` retained as `typedef`
aliases so the 4,590 tests under `regression/python/` keep passing unchanged.
The Python frontend keeps only what is genuinely Python: PEP 484 annotation
handling, the `models/*.py` library, numpy, complex numbers, and the
Python-specific converter dispatch.

This is the answer to the final requirement in the brief: not a new dynamic
representation, but the existing one promoted out of the Python frontend and
given a second consumer.

---

## 6. Runtime modelling

### 6.1 Reused from the Python infrastructure

| Area | Source | Notes |
|---|---|---|
| String storage and methods | `string/` (9.5k lines), `library/python/string.c` | large method overlap; name mapping |
| Array/list storage | `python-list/` (8.3k lines), `library/python/list.c` | infinite-array model carries over |
| Keyed containers | `python-dict/` (4.8k lines) | backs open objects and `Map` |
| Sets | `set/python_set.cpp` | backs `Set` |
| Object allocation | `__ESBMC_new_object` + symex `symex_mem_inf` | reference semantics, non-expiring |
| Exceptions | `cpp-throw`/`cpp-catch`, `remove_exceptions`, `exception_typeid` | shared type-id space |
| Class/struct building | `class/python_class_builder.cpp` | ES6 `class` and shaped objects |
| Adjustment | `clang_cpp_adjust` | reused as-is, as Python does |
| Symbol naming | `symbol_id` (`py:` prefix → `js:`) | one-line change |
| Modules | `module/module_manager.cpp` | resolution strategy differs |
| Math | `library/python/math.c`, `models/math.py` | `Math.*` maps onto it |
| Nondeterminism | `models/nondet.py` → `models/nondet.js` | same intrinsics |
| Intrinsics | `__ESBMC_assume`, `__ESBMC_assert`, `__ESBMC_cover`, `__ESBMC_unreachable` | declared in a JS model file |

### 6.2 JavaScript-specific

| Area | Approach | Milestone |
|---|---|---|
| Prototype chains | `__proto__` field + resolution ladder (§4.4) | 6 |
| `this` binding | conversion-time: method call, `call`/`apply`/`bind`, arrow lexical capture | 6 |
| Coercion (`==`, `+`, `ToPrimitive`) | explicit lowering table; `--js-strict-equality` warns on loose `==` | 4 |
| `ToInt32` bitwise semantics | conversion helper (§4.1) | 4 |
| Array holes / OOB → `undefined` | array OM modification | 5 |
| `Object.keys/values/entries`, spread, destructuring | desugared in the preprocessor | 5 |
| Getters/setters | conversion-time call insertion | 6 |
| Built-in error hierarchy | `models/errors.js` | 3 |
| `Map`/`Set`/`JSON` | `models/*.js` over the dict/set OM | 7 |
| Promises, `async`/`await` | **deferred** — see below | 9 |
| Event loop / microtask queue | **deferred** | 9 |
| Node.js APIs (`fs`, `http`, …) | **deferred**; nondeterministic stubs like `models/os.py` | 9 |
| DOM / browser APIs | **out of scope** | — |

**On async.** Worth stating the shape now even though it is deferred, because it
determines whether the MVP paints itself into a corner. `async`/`await` in a
*single-threaded* event loop is a CPS transformation, not concurrency: an async
function becomes a state machine, `await` a suspension point, and the microtask
queue a deterministic scheduler loop. ESBMC's existing interleaving machinery
(`reachability_tree`, used for pthreads) can drive that scheduler if
non-determinism in resolution order matters. Nothing in the MVP design blocks
it: closures already give the state-machine environments, and exceptions already
give the rejection path.

---

## 7. MVP scope

The target is *computational JavaScript* — algorithmic code, data
transformation, numeric kernels, the kind of thing that appears in coding
benchmarks and in the pure-logic core of real applications. Explicitly not
browser or server behaviour.

### Supported

- **Variables** — `var`/`let`/`const`, hoisting, TDZ, block scope, shadowing.
- **Numbers** — full IEEE-754 double semantics, integral fast path, `ToInt32`
  bitwise operators, `Math.*`.
- **Strings** — literals, template literals with interpolation, concatenation,
  indexing, `length`, and the method set covered by the reused string handler.
- **Booleans, `null`, `undefined`** — including truthiness and `===` vs `==`.
- **Functions** — declarations, expressions, arrow functions, default and rest
  parameters, recursion, first-class function values, **closures with capture**.
- **Conditionals** — `if`/`else`, ternary, `switch` (including fallthrough),
  short-circuit `&&`/`||`/`??`, optional chaining `?.`.
- **Loops** — `while`, `do…while`, C-style `for`, `for…of` over arrays and
  strings, `for…in` over shape-known objects, `break`/`continue` with labels.
- **Arrays** — literals, indexing, `length`, `push`/`pop`/`shift`/`unshift`,
  `slice`/`splice`/`concat`/`indexOf`/`includes`/`join`/`reverse`, and the
  higher-order `map`/`filter`/`reduce`/`forEach`/`find`/`some`/`every` (which
  work precisely *because* closures are in the MVP).
- **Objects** — literals, property read/write, shorthand and computed keys,
  nested objects, `Object.keys/values/entries`, spread and destructuring.
- **Classes** — declarations, constructors, methods, fields, `static`,
  `extends`, `super`, `instanceof`.
- **Exceptions** — `throw`, `try`/`catch`/`finally` including escaping control
  flow, built-in error types, custom errors via `extends Error`.
- **Modules** — ESM `import`/`export` across files.
- **Verification properties** — user `assert`, ESBMC intrinsics, division by
  zero, array bounds, arithmetic overflow on the integral path, uncaught
  exceptions, null/undefined property access, type confusion on unbox.

### Deferred

- `async`/`await`, Promises, the event loop, `queueMicrotask`, timers.
- Generators and iterators (`function*`, `yield`, custom `Symbol.iterator`).
- Proxies, `Reflect`, `Object.defineProperty`, property descriptors, sealing.
- `eval`, `Function` constructor, `with`.
- Regular expressions — Python's `re` model
  (`src/python-frontend/models/re.py`) is a starting point but JS regex is a
  distinct dialect.
- `Map`/`Set`/`WeakMap`/`WeakSet`/`Symbol`/`Date`/`JSON` — post-MVP model files.
- `bigint` beyond the fixed-width approximation.
- Full UTF-16 string semantics.
- Getters/setters (milestone 6, just after MVP).
- CommonJS `require`, Node.js standard library, DOM, Web APIs.
- TypeScript syntax.

---

## 8. Integration with the verification pipeline

The frontend produces a `contextt` and stops. Everything downstream is untouched.

**Registration** — three edits, mirroring Python exactly:

1. `src/langapi/mode.h`: `language_idt::JS`, `new_js_language()`,
   `LANGAPI_MODE_JS`.
2. `src/langapi/mode.cpp`: `extensions_js[] = {"js", "mjs", "cjs", nullptr}`,
   `language_desc_js`, and the `language_desc()` switch arm.
3. `src/esbmc/globals.cpp`: `LANGAPI_MODE_JS` in `mode_table`.

**`js_languaget : languaget`** implements `parse`, `typecheck`, `final`,
`show_parse`, `from_expr`, `from_type` — the same six as `python_languaget`,
and `from_expr`/`from_type` delegate to `c_expr2string`/`c_type2string` for
counterexample rendering exactly as Python does
(`python_language.cpp:379-397`).

**Reuse downstream, with nothing duplicated:**

| Stage | Reuse |
|---|---|
| Symbol management | `contextt`, `symbolt`, `symbol_generator`, `namespacet` |
| Adjustment | `clang_cpp_adjust` (as Python) |
| Operational models | `add_cprover_library()` links the `c2goto` GOTO library |
| IR | `irep2` via `util/irep/migrate.{h,cpp}` at the same seam Python uses |
| GOTO conversion | `goto_convert`, `goto_convert_functions` |
| Exception lowering | `remove_exceptions`, `exception_typeid`, `exception_globals` |
| Property checks | `goto_check` — bounds, division by zero, overflow, pointer safety, all for free |
| Symbolic execution | `goto-symex`, incl. `isinstance2t` resolution and `symex_mem_inf` |
| Slicing, caching | `slice.cpp`, the existing caches |
| SMT | every backend (Z3, Bitwuzla, Boolector, CVC5, MathSAT, Yices, SMT-LIB) |
| Strategies | BMC, incremental BMC, k-induction (modulo the function-pointer caveat), coverage, `--multi-property` |
| Output | counterexamples, witnesses, SARIF, JSON, HTML — all format-agnostic |

**Testing.** `regression/javascript/` in the `REGRESSIONS` list of
`regression/CMakeLists.txt`, with the standard `test.desc` format (line 1
`CORE`/`KNOWNBUG`/`FUTURE`, line 2 source, line 3 flags, line 4+ expected
regexes). `CORE` matters: the Codecov job builds with `CORE_REGRESSION_ONLY=ON`,
so only `CORE` tests move coverage.

**Explicitly avoided duplication:** no second IR, no second symbol table, no
second exception mechanism, no second container model, no second string model,
no second class model, and no frontend-local property checking.

---

## 9. Architecture

### 9.1 Components

```
 ┌──────────────────────────────────────────────────────────────┐
 │ src/js-frontend/                              (new, JS-only) │
 │                                                              │
 │  libs/acorn/acorn.js            vendored parser (MIT)        │
 │  parser/parse.js                driver → ESTree JSON         │
 │  parser/preprocess.js           desugaring: destructuring,   │
 │                                 spread, for-of, optional     │
 │                                 chaining, closure conversion │
 │  parser/annotate.js             scope + type inference (§5.2)│
 │  models/*.js                    errors, nondet, intrinsics,  │
 │                                 Math, Map/Set/JSON           │
 │                                                              │
 │  js_language.{h,cpp}            languaget implementation     │
 │  js_converter.{h,cpp}           ESTree JSON → contextt       │
 │  converter/                     expr, stmt, funcdef, funcall,│
 │                                 class, binop, member         │
 │  type/js_type_handler.{h,cpp}   ESTree annotations → typet   │
 │  js_object/                     shape inference, prototypes  │
 │  js_closure/                    environment materialisation  │
 └───────────────────────────┬──────────────────────────────────┘
                             │
 ┌───────────────────────────▼──────────────────────────────────┐
 │ src/dynlang/                         (new, from python-*/)   │
 │   dyn_value/  dyn_container/  dyn_string/  dyn_object/       │
 │   dyn_exception/  dyn_scope/                                 │
 └───────────────────────────┬──────────────────────────────────┘
                             │
 ┌───────────────────────────▼──────────────────────────────────┐
 │ src/c2goto/library/dyn/*.c    DynObject, DynArray, DynDict,  │
 │                               string ops   (from python/*.c) │
 └───────────────────────────┬──────────────────────────────────┘
                             │
 ┌───────────────────────────▼──────────────────────────────────┐
 │ contextt → clang_cpp_adjust → goto_convert → goto-symex → SMT│
 └──────────────────────────────────────────────────────────────┘
```

### 9.2 Data flow

```
 main.js
   │ js_languaget::parse()
   ├─▶ [Stage 1] node parse.js      ‖  [Stage 2] embedded QuickJS-ng
   │       acorn.parse(src, {ecmaVersion:"latest", sourceType:"module",
   │                         locations:true})
   ├─▶ preprocess.js   desugar to a small core; lambda-lift closures
   ├─▶ annotate.js     scopes, type lattice fixpoint, narrowing,
   │                   object-shape inference; write annotations onto nodes
   ├─▶ import resolution: recurse into imported modules and models
   ▼
 <module>.json  (annotated ESTree)
   │ js_languaget::typecheck()
   ├─▶ add_cprover_library(context)        ← dyn OM + C library
   ├─▶ js_converter::convert()
   │       ├─ monomorphic node → concrete typet, direct irep2 term
   │       ├─ polymorphic node → DynObject box + narrowed dispatch
   │       ├─ shaped object    → struct_typet via dyn_object
   │       ├─ open object      → dyn_container dict
   │       ├─ closure          → {fn, env}, indirect call
   │       └─ throw/try        → cpp-throw / cpp-catch
   ├─▶ clang_cpp_adjust
   ▼
 contextt → goto_convert → … → VERIFICATION SUCCESSFUL / FAILED
```

### 9.3 Reuse points, named

| JS component | Reuses | Kind |
|---|---|---|
| `js_language.cpp` | `python_language.cpp` structure | copy + adapt |
| `parser/parse.js` | `parser/parser.py` structure, FLAIL mangling | copy + adapt |
| `parser/preprocess.js` | `preprocessor/*.py` mixin design | design reuse |
| `parser/annotate.js` | `python_annotation` JSON-annotation contract | design reuse |
| `js_converter` | `python_converter` dispatch skeleton | copy + adapt |
| `js_type_handler` | `type_handler` (`get_typet`, optional types) | copy + adapt |
| arrays | `python-list/` + `library/python/list.c` | **shared** |
| open objects, `Map` | `python-dict/` | **shared** |
| strings | `string/` + `library/python/string.c` | **shared** |
| `Set` | `set/python_set.cpp` | **shared** |
| shapes, classes | `class/python_class_builder.cpp` | **shared** |
| exceptions | `exception/` + `remove_exceptions` | **shared** |
| modules | `module/module_manager.cpp` | **shared** |
| naming | `symbol_id` | **shared** |
| tags | `PyObject` → `DynObject` | **shared** |
| `typeof`/`instanceof`/`in` | `isinstance2t`/`hasattr2t`/`isnone2t` + symex | **shared** |

### 9.4 New JavaScript-specific components

Only five: the Acorn integration, the type-inference pass, object-shape
inference and prototype resolution, closure conversion, and the coercion tables
(`ToInt32`, `ToPrimitive`, `ToBoolean`, `==`). Everything else is shared or
adapted.

---

## 10. Roadmap

Each milestone is independently mergeable, keeps `regression/python/` green, and
ships at least one passing and one failing regression test per the project's PR
convention.

### M1 — Acorn integration and language registration

**Deliverables.** `js_languaget` registered for `.js`/`.mjs`/`.cjs`; Acorn
vendored under `src/js-frontend/libs/acorn/` and FLAIL-mangled; `parse.js`
driver; `--parse-tree-only` dumps ESTree JSON; Node discovery and version check.
**Risks.** Node on `PATH` (accepted; M7 removes it). FLAIL asset layout for a
multi-file JS bundle.
**Testing.** Round-trip a corpus of ~50 JS files through `--parse-tree-only` and
diff against `node -e "acorn.parse(...)"`.
**Validation.** Every file parses; JSON is byte-identical to a direct Acorn run.

### M2 — Skeleton converter: expressions, statements, `main`

**Deliverables.** `js_converter` walking ESTree; numeric literals, arithmetic,
comparison, `let`/`const`/`var`, `if`, `while`, `for`; `assert` and the ESBMC
intrinsics; program entry synthesis.
**Risks.** Getting the `--ir`/rounding-mode discipline right from the start
(`python_language.cpp:271-279` documents the failure mode: folded constants
drifting one ulp and flipping verdicts).
**Testing.** `regression/javascript/` bootstrapped, ~30 tests.
**Validation.** `esbmc t.js` reports `VERIFICATION SUCCESSFUL`/`FAILED`
correctly on straight-line and looping arithmetic; division-by-zero and
overflow properties fire.

### M3 — Shared dynamic-language layer extraction

**Deliverables.** `src/dynlang/` and `src/c2goto/library/dyn/` created by moving
Python components; `PyObject`/`PyListObject` retained as typedefs; Python
frontend rebuilt against the shared layer; `models/errors.js`, `models/nondet.js`.
**Risks.** *This is the highest-risk milestone* — it touches code covered by
4,590 tests. Mitigation: pure relocation, no behaviour change, one component per
commit, full `regression/python` after each.
**Testing.** `regression/python` must be bit-identical before and after; capture
verdicts to a baseline file and diff.
**Validation.** Zero Python regressions; the JS frontend links only against
`src/dynlang/`, never `src/python-frontend/`.

### M4 — Type inference, coercion, and the tagged path

**Deliverables.** `annotate.js` (scopes, lattice, fixpoint, narrowing);
`js_type_handler` consuming the annotations; integral-number fast path;
`ToInt32`/`ToPrimitive`/`ToBoolean` tables; `===` vs `==`; **widened
`dyn_assign`/`dyn_binop`/`dyn_unbox`** lifting the literal-only restriction.
**Risks.** Inference precision decides performance — an imprecise pass tags
everything and the frontend is unusably slow. Unsound narrowing silently drops
paths.
**Testing.** Inference unit tests (Catch2, over annotated JSON); a
tagged-vs-monomorphic benchmark set; **differential testing against Node**:
run each test under `node` and compare the observable result with ESBMC's
verdict, the analogue of `scripts/check_python_tests.sh`.
**Validation.** ≥90% of bindings in the benchmark corpus infer monomorphic;
zero differential mismatches; the widened tagged path passes tests that
currently throw "not yet supported" on the Python side.

### M5 — Arrays and objects

**Deliverables.** Array literals, indexing, `length`, mutation and query methods
over the shared array OM; OOB→`undefined`; object literals with shape inference
→ `struct_typet`; open objects → dict; destructuring and spread desugaring;
`Object.keys/values/entries`.
**Risks.** Shape inference misjudging an object as closed when a later branch
adds a key — must be conservative, and the desugaring must be verified against
Node.
**Testing.** ~120 regression tests; property-based comparison against Node for
array method semantics.
**Validation.** Bounds properties fire correctly; a shaped object costs the same
as the equivalent C struct in VCC count.

### M6 — Classes, prototypes, `this`, closures

**Deliverables.** ES6 classes via `python_class_builder`; `extends`/`super`;
`instanceof`; prototype resolution ladder (§4.4); `this` binding incl. arrow
lexical capture and `call`/`apply`/`bind`; getters/setters; **closure conversion**
with by-reference capture of reassigned bindings; devirtualisation of
statically-unique callees.
**Risks.** By-reference capture is the classic `for (var i…)` trap and must be
exactly right. Function-pointer calls disable k-induction — measure how often
devirtualisation avoids that.
**Testing.** Closure-capture tests including the loop-variable case;
`map`/`filter`/`reduce`; inheritance chains; a k-induction subset over
devirtualised code.
**Validation.** `[1,2,3].map(x => x*2)` verifies; capture semantics match Node
on the full closure test set; devirtualisation covers ≥80% of call sites in the
corpus.

### M7 — Runtime library and embedded parser

**Deliverables.** `models/*.js` for `Math`, `JSON`, `Map`, `Set`, `Number`,
`String` statics, the error hierarchy; ESM import resolution via
`module_manager`; **QuickJS-ng embedded**, `--js-parser=node|embedded`, Node
dependency optional.
**Risks.** QuickJS-ng build integration across Linux/macOS/Windows and the
`DOWNLOAD_DEPENDENCIES` flow. Keep Stage 1 supported so a build problem is never
a blocker.
**Testing.** Per-model regression tests; parity tests asserting both parser
backends produce identical JSON on the whole corpus.
**Validation.** ESBMC verifies a multi-module JS program with no `node` on
`PATH`; both backends agree byte-for-byte.

### M8 — Benchmarks and evaluation

**Deliverables.** A `regression/javascript-intensive/` suite mirroring
`regression/python-intensive/`; a HumanEval-JS style corpus (the repo already
carries `regression/humaneval`); comparison against the Python frontend on
algorithms implemented in both languages; performance and coverage reporting.
**Risks.** Selection bias — a corpus tuned to what the frontend supports proves
nothing. Draw from an external, pre-existing corpus.
**Testing.** Full suite under the project's 5-minute cap, scoped per-suite.
**Validation.** Published pass rate, per-stage timing, and an explicit
unsupported-construct list with counts.

### M9 — Advanced features

**Deliverables, in dependency order.** Generators and iterators (state-machine
transform, reusing closure environments) → `async`/`await` and Promises (CPS +
microtask queue, optionally scheduled by `reachability_tree`) → regular
expressions (JS dialect, starting from `models/re.py`'s approach) → full UTF-16
strings → true bignum (shared with Python's `FIXME`) → nondeterministic Node API
stubs in the style of `models/os.py`.
**Risks.** Each is independently large; sequence them by demand from M8's
unsupported-construct counts rather than by a predetermined order.
**Testing.** Per-feature suites; async tests must pin scheduling determinism.
**Validation.** Feature-by-feature, against Node's observable behaviour.

**Sequencing note.** M3 and M4 are the load-bearing milestones and also the
riskiest. M1, M2 and M3 can be worked in parallel by separate contributors once
M1's JSON contract is fixed; M5 and M6 both depend on M4's annotations.

---

## 11. Why not a new dynamic-language representation

The brief asks any proposal introducing a new representation to justify it.
This proposal introduces none, for four reasons.

1. **The tag layout is already right.** `{value, float_idx, type_id, size}`
   carries an untyped payload pointer, a type tag, a width, and an
   encoding-mode escape hatch. JavaScript needs exactly that. The only changes
   are new tag values (`undefined`, `symbol`) in an already-open namespace.

2. **The expensive parts are already built and already tested.** 26k lines of
   C++ across the list, dict, string, set, tuple, class and exception handlers,
   plus 4.2k lines of C operational models, plus 4,590 regression tests. A
   parallel JS-only representation would mean reimplementing all of it *and*
   maintaining two divergent models of the same abstractions.

3. **The seams are already language-neutral.** `isinstance2t`/`hasattr2t`/
   `isnone2t` are irep2 nodes with symex lowerings that query the points-to
   set — they know nothing about Python. `__ESBMC_new_object`'s symex
   interception, the infinite-array container trick, the `cpp-throw` reuse and
   `clang_cpp_adjust` are all generic mechanisms that happen to have been
   reached through Python first.

4. **The one real gap is a gap in the Python frontend too.** Non-literal
   assignment to a tagged variable throws today
   (`converter_stmt.cpp:3243`). Fixing that in a shared layer fixes both
   languages; forking it fixes one and leaves the other with a
   `runtime_error`. The same argument applies to bignum, to UTF-16, and to
   `try`/`finally` with escaping control flow — three more places where the JS
   requirement is the forcing function for a Python fix.

Where JavaScript genuinely differs — prototype chains, capturing closures, the
`ToInt32` bitwise rules, `undefined` as distinct from `null`, `==` coercion —
the proposal adds *JavaScript-specific components on top of the shared layer*,
not a competing substrate. That boundary is what §9.4's list of five new
components is: the irreducible JavaScript-specific surface, and nothing more.
