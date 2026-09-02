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
- **The runtime substrate is reusable after generalisation — behind an
  interface, not by relocation.** `PyObject`, `PyListObject`, the string
  handler, the dict handler, the exception lowering and the
  class-to-`struct_typet` builder become a shared *dynamic-language layer*, but
  only once each sits behind a stated contract (§5.5) and the extraction's entry
  criteria are met (§10, M3). Method resolution is explicitly *not* shared:
  Python's MRO and JavaScript's prototype chain are different algorithms, and
  each frontend keeps its own resolver. Anything that cannot be stated without a
  language conditional is duplicated instead of shared.
- **The typing strategy is not reusable as-is, and this is the crux of the
  proposal.** ESBMC's Python frontend is, in practice, a static
  monomorphising translator that leans on PEP 484 annotations; its genuinely
  dynamic path accepts an rvalue only when that rvalue is a numeric scalar or a
  string, and raises "not yet supported" for anything else — a container, a
  class instance, a call whose result type is unknown
  (`src/python-frontend/converter/converter_stmt.cpp:3059-3076`). JavaScript has
  no annotations. The frontend therefore needs a real inference pass and a
  first-class tagged-value path — and building that is the one large piece of
  new infrastructure, which pays back into Python immediately.

Everything below is evidence for those three claims and a plan that follows from
them. §1.5 states the strongest objection to the whole approach — that the
Python frontend is too unsettled to build a second language on — and the three
constraints the rest of the document is written to respect because of it.

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
    │  preprocessor/*.py (18 mixins, ~11k lines) desugars in Python
    │  python_annotation<json>      adds type nodes to the JSON  (:243)
    ▼
  annotated JSON AST
    │  python_converter::convert()                     python_converter.cpp
    │    ~68 translation units: converter/, python-list/, python-dict/,
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
becomes `PyObject`-typed when the tag analysis flags it, and assignment into a
tagged variable is gated on the *type of the rvalue*:

```cpp
if (dynamic_type_handler_.is_tagged(name)) {
  if (ast_node.contains("value") && !ast_node["value"].is_null()) {
    exprt rhs = get_expr(ast_node["value"]);
    if (type_handler_.is_numeric_scalar_type(rhs.type()) ||
        type_handler_.is_string_type(rhs.type())) {
      dynamic_type_handler_.assign(rhs, …); return;
    }
  }
  throw std::runtime_error(
    "assigning a value of this type to a dynamically-typed variable "
    "is not yet supported");
}
```
— `src/python-frontend/converter/converter_stmt.cpp:3059-3076`

So the boundary is *by rvalue type*, not by literal-vs-computed: a computed
numeric or string rvalue does box, while a container, a class instance, a tuple
or a call whose result type is unknown throws. `is_numeric_scalar_type`
additionally excludes bitvectors narrower than 16 bits
(`type/type_handler.cpp`).

That is the honest state of the art, and it is the shape JavaScript cannot live
within: in JS the *common* tagged rvalue is an object, an array or a call
result — exactly the three the check rejects.

**Classes reuse the C++ object model.** `python_class_builder::build`
(`src/python-frontend/class/python_class_builder.cpp`) emits a `struct_typet`
with a method table and base classes, instances are allocated through
`__ESBMC_new_object`, and `clang_cpp_adjust` — the C++ frontend's adjuster —
runs over the result (`python_language.cpp:314`).

The allocation mechanism is worth naming precisely, because §6.1 and §9.3 rest
on reusing it. `__ESBMC_new_object`'s C body is a placeholder returning `0`
(`src/c2goto/library/python/list.c:14`); the real work is a dedicated handler in
symex (`src/goto-symex/symex_main.cpp:1058-1116`) that allocates one dynamic
object typed by the callee's *result pointer type*, marks it non-expiring so it
survives the constructing frame, and binds the return value to its address. It
is **not** `symex_mem_inf`. That function (`builtin_functions/memory_alloc.cpp:482`)
is a different allocator, reached only through the `__ESBMC_create_inf_obj`
handler (`symex_main.cpp:1118`), and it produces the *infinite array* backing a
list's `items` — which is why §4.2.1 can reuse it for JS arrays while class
instances go through the size-1 path instead. The `__ESBMC_new_object` handler's
own comment describes itself as mirroring `symex_mem_inf`'s binding, and the
comment at `src/c2goto/library/python/list.c:11` still names `symex_mem_inf`
outright; that comment is stale and predates this document.

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

### 1.5 The objection this design has to answer first

The Python frontend is not a settled component. It is under active development,
its dynamic path is the narrow thing §1.3 just described, and several of its
subsystems are still changing shape. Building a second frontend on top of it,
and then *promoting its internals to a shared interface*, risks freezing
immature abstractions into a contract two languages depend on — and a shared
layer that both frontends can edit is a faster route to entanglement than two
separate frontends would be.

This is the strongest argument against the proposal and it is not answered by
asserting that reuse is cheaper. It is answered by three constraints, which the
rest of this document is written to respect:

1. **Reuse is by interface, not by relocation.** §5.5 defines an interface
   before any code moves; nothing is shared because it happens to be sitting in
   `src/python-frontend/`. A component qualifies for the shared layer only if it
   has a stated contract, its Python-specific policy has been factored out of
   it, and a second consumer actually exercises it. Components that fail that
   test stay in the Python frontend and are *duplicated* on the JS side — a
   deliberate accepted cost, because two clear implementations beat one
   entangled one.
2. **The dependency is one-way and enforced.** `src/js-frontend/` must not
   include anything from `src/python-frontend/`, and vice versa; both include
   `src/dynlang/`. This is checkable in CI (an include-path lint), and it is
   what stops "shared" from degrading into "mutually coupled".
3. **The extraction is gated, not scheduled.** M3 does not start because the
   roadmap says it is next. Its entry criteria are in §10, split across M3a (the
   interface exists and the Python frontend has been rebuilt against it) and M3b
   (a JavaScript consumer actually calls it); if they are unmet, the JS frontend
   proceeds on duplicated code and the extraction is deferred.
   A design whose first large milestone is a refactor of someone else's
   experimental subsystem should be able to survive that refactor not happening.

There is a real benefit in the other direction that the objection should be
weighed against: every gap in §1.4 is a gap JavaScript *forces*, and the four
listed there (dynamic scalars, iteration, closures, `try`/`finally`) are the
Python frontend's own known weaknesses. A second consumer is the usual way an
experimental component acquires a stable interface. But that argument only
holds under constraint (1); without it, the objection stands.

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

**What Acorn does not give you.** It produces an AST and nothing else: no scope
tree, no binding table, no type information. Scope resolution is a separate
analysis the annotation pass owns (§5.2), and the recommendation there is to
vendor `eslint-scope` for it. So the vendored bundle is Acorn + `acorn-walk` +
`eslint-scope` (+ `esrecurse`/`estraverse`), all MIT or BSD-2 and all pure JS.
"Zero dependencies" above is a property of Acorn itself and of the *native*
toolchain — which is what §2.4's embedding argument actually rests on — not a
claim that the frontend vendors exactly one file.

### 2.3 Why not the others

- **SWC.** The performance argument does not survive the interface. Published
  benchmarking of Rust JS parsers consistently finds that serialising the Rust
  AST across the boundary eats most of the parse-time win — and ESBMC *must*
  cross that boundary, because the converter is C++. Against that we would take
  on a Rust toolchain in ESBMC's build, a non-ESTree AST, and a second
  dependency-management story. Parse time is not ESBMC's bottleneck: the Python
  frontend spends ~0.6–1.2 s in "GOTO program creation" and milliseconds in the
  solver on small programs (`src/python-frontend/README.md`, Examples 2 and 6:
  0.637 s and 1.156 s).
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

**Contract between stages.** The contract is the *ESTree AST*, not its
serialisation: both stages must produce structurally identical trees — same node
kinds, same children, same `start`/`end`/`loc` — so the converter is unaware of
which is in use, and Stage 1 remains a supported fallback
(`--js-parser=node|embedded`).

Byte-identical JSON is deliberately *not* the requirement. `JSON.stringify` is
free to differ between Node's serialiser and QuickJS-ng's in key order, number
formatting (`1e21` vs `1000000000000000000000`, `-0`), and non-ASCII escaping,
and pinning the contract to text would make an unrelated engine upgrade look
like a parser regression. Equality is therefore defined on the parsed tree, by a
structural comparison that ignores key order and normalises numeric literals.

Where a *textual* artefact is genuinely needed — golden-file tests, and the
cross-backend parity check in M7 — it is produced by one canonical serialiser
that both backends feed (recursive key sort, fixed number formatting, `\u`
escaping above ASCII), so the canonical form is a property of the test harness
rather than of either engine.

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
| `undefined` | *(none)* | sentinel, own pointee type (§4.1) | — | `isnone2t` variant | **N** (small) |
| `symbol` | *(none)* | opaque unique id | — | `unsignedbv` | **N** (deferred) |
| array | `list` | `JsArrayObject` (+ presence, `length` invariant) | `list.c` | infinite `array` + `index` | **G** (§4.2.0) |
| object (fixed shape) | class instance | `struct_typet` | `__ESBMC_new_object` | `member`, `with` | **R** |
| object (dynamic keys) | `dict` | `DynDict` | `python-dict/` | struct + arrays | **G** |
| class | class | `struct_typet` + methods | `python_class_builder` | `member`, indirect call | **R** |
| inheritance | inheritance | base subobject | `get_bases`, `clang_cpp_adjust` | `member` chain | **R** |
| prototype chain | *(none)* | `__proto__` link | new OM | convert-time walk / OM loop | **N** |
| function | function | function symbol | — | `code_function_call` | **R** |
| closure | nested function | `{fn, env}` pair | `__ESBMC_new_object` env | fn-pointer call | **N** (see §4.3) |
| `this` | `self` (bound at definition) | call-site–determined receiver | — | leading param + `undefined` sentinel | **N** (§4.6) |
| exception | exception | `cpp-throw`/`cpp-catch` | `remove_exceptions` | `code_cpp_throw` | **R** |
| `finally` | `finally` (restricted) | — | — | — | **G** |
| dynamic typing | annotations + narrow tags | `DynObject` | shared tag OM | `isinstance2t` | **G** — the big one |
| `for...of` | `for … in range()` | desugared | — | `code_goto` loop | **G** |
| modules (ESM) | `import` | module manager | `module_manager` | — | **G** |
| `typeof` | `type(x)` | tag read + spec result table | — | `isinstance2t` | **R** (`null`/`function` caveat) |
| `instanceof` | `isinstance` | value-set query | `python_builtins.cpp` | `isinstance2t` | **R** |
| `in` | `in` (dict) | key membership | `python-dict/` | `hasattr2t` | **R** |

Aggregate: of 24 constructs, 12 reuse directly, 7 generalise a Python component,
and 5 are new. Two of the five are small (`undefined`, `symbol`); the other three
are real work — prototype chains, closures, and `this` binding, which is **N**
rather than **R** because JavaScript determines it at the *call* site while
Python's `self` is fixed when the method is looked up (§4.6, and §6.2 schedules
it at M6).

The single largest item is not in the **N** column at all: dynamic typing is
marked **G**, because it widens the existing Python tagged-value path rather than
introducing a new one (§5.3) — but it is still the biggest piece of work in the
plan, and §9.4 counts it accordingly.

---

## 4. Mapping JavaScript into irep2

### 4.1 Primitives

**`number`.** One type, IEEE-754 binary64: `floatbv(52,11)`, the same
`double_type()` the Python frontend produces for `float`. Arithmetic maps to
`ieee_add`/`ieee_sub`/`ieee_mul`/`ieee_div` with the rounding-mode operand, and
`Number.isNaN`/`isFinite` map to `isnan2t`/`isfinite2t`.

Two JavaScript-specific obligations:

- **Bitwise operators are `ToInt32`, which is not a typecast.** `a | 0`,
  `a << 1`, `a & 0xff` and `~a` all convert through `ToInt32` (ECMA-262,
  `sec-toint32`), a *total* function on doubles:

  1. NaN, `+∞`, `−∞`, `+0` and `−0` all map to `+0`.
  2. Otherwise, truncate toward zero.
  3. Reduce the result modulo 2³², then wrap into the signed range: a residue
     `≥ 2³¹` becomes `residue − 2³²`.

  So `4294967296 | 0` is `0`, `2147483648 | 0` is `-2147483648`, `NaN | 0` is
  `0`, and `1e300 | 0` is `0`. `ToUint32` (`sec-touint32`) is steps 1–3 without
  the final wrap, and is what `>>>` uses.

  A bare `typecast(floatbv → signedbv(32))` implements none of this. It has no
  modulo step, so every out-of-range operand is wrong rather than merely
  imprecise, and SMT-LIB leaves `fp.to_sbv` *unspecified* for NaN, infinities
  and out-of-range values, so the solver is free to pick any result — the
  encoding would be unsound in a way no test reliably catches. The lowering is
  therefore an explicit helper, in this order: guard NaN and the infinities to
  `+0`; truncate toward zero; take the residue modulo 2³² **in `floatbv`**,
  because the truncated value need not fit in any fixed-width integer
  (`1e300` is an exact double); only then convert the in-range residue to
  `unsignedbv(32)` and reinterpret as `signedbv(32)`. The bit operation applies
  to that, and the result converts back to `floatbv`.

  Getting the order wrong — casting first and reducing afterwards — is exactly
  the bug this paragraph exists to prevent. It is a small, self-contained
  helper, and it is the single most common source of unsoundness in naive JS
  models, so it belongs in milestone 4, not later.
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
`None`. Both are singleton sentinels, and the sentinel's *pointee* type is what
makes them distinguishable on the monomorphic path — so it has to be a type
nothing else in the frontend uses.

The obvious spelling, `pointer(char)`, does not work: that is exactly the type
§4.1 gives a computed string, so `typeof s === "undefined"` and "`s` is a string
pointer" would be the same type-level question, and `string | undefined` — one
of the most common shapes in real JS — would be unrepresentable without boxing.
Two of the other short candidates are taken as well: `pointer(bool)` is
`none_type()` and `pointer(empty)` (i.e. `void *`) is `any_type()`
(`src/util/lang/python_types.cpp:4-15`).

So `undefined` gets a pointee type that exists only to be that sentinel:

```cpp
typet js_null_type() { return none_type(); }   // pointer(bool), reused as-is

typet js_undefined_type()
{
  // Empty tag-only struct: distinct from bool (null), void (Any) and char
  // (computed string). The pointer is a sentinel address and is never
  // dereferenced, so the struct having no members is fine.
  struct_typet t;
  t.tag("__ESBMC_js_undefined");
  return pointer_typet(t);
}
```

with distinct `type_id`s in the shared tag namespace, so `x === null` and
`x === undefined` are separate predicates while `x == null` (loose) is their
disjunction. Sentinel values are compared by identity, never dereferenced; any
read through one is a `TypeError` property, which is what makes
"`undefined` is not a function" a *reported* defect rather than a crash.

On the monomorphic path a variable that may be `undefined` uses the optional-type
builder Python already has (`type_handler::build_optional_type`,
`type/type_handler.cpp:1451`), so `string | undefined` is an optional over
`char *` rather than a union of two pointer types. Where inference cannot keep a
binding monomorphic, `undefined` is carried on the tagged path as its own
`type_id` and the sentinel type never appears — the sentinel exists only so the
*monomorphic* path can still tell the two apart.

**`symbol`.** Deferred past the MVP. When needed: an opaque `unsignedbv(64)`
identity allocated by a counter, with `Symbol()` calls yielding fresh distinct
values, and well-known symbols as reserved constants.

### 4.2 Objects and arrays

The single most important modelling decision, because it decides whether
verification is fast or hopeless.

#### 4.2.0 What the ECMAScript object model actually is

Stating this first, because the two tiers below are *approximations of it*, and
a reader cannot judge an approximation without the thing being approximated.
Python's `list` and `dict` are close enough to their JS counterparts to invite
a one-to-one mapping, and that mapping is wrong in five specific ways.

**1. There is one data structure, and it is an ordered property map.** Every
JS object is a map from property keys to property *descriptors* (value or
get/set, plus `writable`/`enumerable`/`configurable`), with a `[[Prototype]]`
link. There is no separate "array type" and no separate "record type" — an
array is an object, a class instance is an object, a function is an object.
Python's `list`/`dict`/instance split does not exist here, so a design that
maps JS array → `PyListObject` and JS object → `PyDict` has silently asserted a
partition JavaScript does not have.

**2. Keys are strings (or symbols); integer indices are canonicalised.**
`a[1]`, `a["1"]` and `a[1.0]` are the same property. `a[1.5]` and `a["01"]` are
ordinary string keys, *not* indices. An array index is specifically a string
that round-trips through `ToString(ToUint32(k))` and is `< 2³²−1`; everything
else is a normal key that does not participate in `length`. The model needs
this canonicalisation at every computed access, and it is a place where a naive
`index2t` on a numeric key is unsound for `a[1.5]`.

**3. Arrays are exotic only in `length`.** The one non-ordinary behaviour is
that `length` is a data property whose `[[DefineOwnProperty]]` is magic: writing
an index `≥ length` raises `length`, and writing a smaller `length` *deletes*
every index above it. Everything else about an array — `a.foo = 1` is legal and
does not touch `length`; `delete a[0]` leaves `length` alone — falls out of (1).
So `JsArrayObject` cannot be `PyListObject` with a renamed field: it needs the
`length`/index coupling as an invariant, and it needs somewhere to put
non-index properties.

**4. Holes are not `undefined`, and the difference is observable.** `[1,,3]`
has no property `"1"`; `[1,undefined,3]` has one whose value is `undefined`.
`1 in a`, `Object.keys(a)` and `a.hasOwnProperty(1)` distinguish them, and the
array methods split three ways: `forEach`/`filter`/`every`/`some` skip holes,
`map` skips them but *preserves* them in the result, and `Array.from`, spread,
`entries`/`keys`/`values`, `find`/`findIndex` treat a hole as `undefined`. A
model with a flat `items` buffer and no presence bit collapses this distinction
and will disagree with Node on any sparse-array test.

**5. Property order is specified, and programs depend on it.** Within one
object, own keys come out as: array-index keys in ascending numeric order,
then remaining string keys in property-creation order, then symbols in creation
order. `Object.keys`, `Object.values`, `Object.entries`, `JSON.stringify` and
`for...in` all follow it (`for...in` then repeats per prototype-chain
component, and its order becomes unspecified only if the object is mutated
during iteration). So insertion order is *semantics*, not an implementation
detail: an unordered map model gives wrong answers for any program that
serialises or enumerates an object.

**Consequences for the two tiers.** Tier 1 (`struct_typet`) is sound only for
objects whose key set *and* creation order are both fixed at the allocation
site, which is why shape inference has to be conservative about any computed or
conditional key. Tier 2 must carry a presence bit and a creation-order index
per entry, not just key→value; Python's dict handler is insertion-ordered
already (CPython semantics), so this is a reuse point rather than new work, but
it has to be *checked* rather than assumed. Arrays get an explicit
`present[]` bitmap alongside `items[]` so holes survive, and `length` is
maintained as an invariant rather than derived.

None of this changes the tier choice below; it changes what each tier has to
store, and it is the reason `JsArrayObject` is not simply `PyListObject`.

#### 4.2.1 The two tiers

**Arrays.** Generalise `PyListObject` to

```c
typedef struct __ESBMC_JsArrayObj {
  DynType   *type;
  DynObject *items;    /* infinite array, materialised by symex          */
  _Bool     *present;  /* infinite array: hole vs. present (§4.2.0 #4)   */
  size_t     length;   /* coupled to items by the invariant below        */
  DynDict   *props;    /* non-index own properties, NULL until used (#3) */
} JsArrayObject;
```

which is `PyListObject` plus the two fields §4.2.0 forces. The infinite-array
trick (`__ESBMC_create_inf_obj`) carries over unchanged and is precisely what
makes symbolic indices tractable; `present` is a second infinite array, so holes
cost one extra bit per touched index and nothing for arrays that have none —
the annotation pass marks an array *dense* when no hole can reach it, and the
`present` reads fold away.

JS-specific behaviour that Python's list lacks:

- **OOB read yields `undefined`**, it does not raise. This is the language's
  semantics, and it means `goto_check`'s array-bounds property does not carry
  over to JS index access the way §8's "for free" row suggests for C: the access
  is in bounds of the infinite `items` array *by construction*, so `goto_check`
  has nothing to fire on, and the thing a user might want reported — that the
  index was outside `[0, length)` — is a JS-level property the array OM has to
  raise itself. The MVP therefore defaults to `--js-array-oob=undefined`
  (faithful to the language) with `--js-array-oob=error` reporting every
  out-of-range index as a violated property. **The default is deliberately
  permissive about a class of defect ESBMC would normally report**, which is why
  §7 lists array bounds as opt-in rather than as a default property.
- **`length` is an invariant, not a derived value.** Writing index `i ≥ length`
  sets `length = i+1`; writing `length = n` clears `present[k]` for all
  `k ≥ n`. Both directions are maintained by the OM.
- **Non-index properties do not touch `length`**, which is why `props` exists;
  it stays `NULL` for the overwhelmingly common array-used-as-array case, so it
  costs nothing until a program actually writes `a.foo`.

Each is a modification to the array OM rather than a new model, but `present`
and `props` are additions to the *layout*, so this is a generalisation of
`PyListObject` and not a rename.

**Objects — two tiers, chosen per allocation site.**

*Tier 1: shape-inferred objects → `struct_typet`.* When the annotation pass can
prove an object's key set is fixed (an object literal whose properties are never
added to or deleted, which covers the overwhelming majority of computational
JS), emit a `struct_typet` exactly as `python_class_builder` does for a class
instance. Then `o.x` is `member2t`, `o.x = v` is `with2t`/`code_assign`, and the
whole thing costs the solver what a C struct costs. Instances are allocated by
`__ESBMC_new_object` so they get reference semantics and survive escaping their
defining scope — the mechanism Python class instances already use.

Member order in the emitted struct is the §4.2.0 #5 enumeration order, so
`Object.keys` and `JSON.stringify` are a walk over `member_names` and need no
side table. The inference must be conservative in *both* directions: a key that
might be added on some path, and a key whose creation order differs between
paths, both disqualify the site from Tier 1.

*Tier 2: open objects → `DynDict`.* When keys are computed (`o[k]`), added, or
deleted, fall back to the generalised dict handler (`src/python-frontend/
python-dict/`, ~4.8k lines across 8 translation units) keyed by string with
`DynObject` values. Slower, general, already written — but it has to be
*confirmed* insertion-ordered and given a presence bit before it can carry
JS objects (§4.2.0 #4, #5); CPython dict semantics make the first likely and
the second is new either way. Index-like keys additionally need the ascending
numeric ordering of #2, which a plain insertion-ordered dict does not give: the
key set is partitioned into index keys and string keys, and enumeration
concatenates the two.

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

1. Compute free variables per function, over the scope tree the annotation pass
   builds (§5.2 — Acorn does not emit one; `acorn-walk` supplies the traversal).
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
2. **Known shape, inherited property, prototype chain provably frozen at this
   site** → walk the `__proto__` chain *at conversion time* and emit `member2t`
   on the resolved subobject. This is exactly what C++ does for base-class
   members, and `clang_cpp_adjust` already implements the subobject arithmetic.

   The third condition is a real precondition, not a formality, and it is *not*
   the common case by default. §5.5 is right that a `[[Prototype]]` chain is
   mutable at runtime: `Object.setPrototypeOf`, reassigning `Foo.prototype`, and
   monkey-patching a method onto a prototype after instances already exist are
   all ordinary JavaScript, and mixin and polyfill code is built out of them.
   Baking a resolution into a `member2t` without proving the chain is stable
   would silently miscompile that code — the worst failure mode a verifier has.

   So the annotation pass must discharge, for every shape the receiver may
   have: no reachable `Object.setPrototypeOf` or `Reflect.setPrototypeOf` on it,
   no write to any `.prototype` property on its constructor, and no write to a
   property of any object on its chain, on any path that can execute before this
   site. Like Tier-1 shape inference in §4.2.1, this must be conservative in
   *both* directions — an unresolvable callee, a computed property write whose
   target set includes a possible prototype, or any escape of a prototype object
   into an unanalysed context disqualifies the site.

   A disqualified site is not a failure and is not out of scope: it falls to
   step 5 when the shape set is finite, and to step 6 otherwise. Prototype
   reassignment is therefore *handled* in the MVP — it is simply handled by the
   dynamic steps of this ladder rather than by this one.
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

### 4.6 `this`

Python's `self` is an ordinary leading parameter, bound when the method is
resolved on the class. JavaScript's `this` is supplied by the *call*, which is
why §3 marks it **N** and not **R**:

| Call form | `this` |
|---|---|
| `obj.m(…)` | `obj` |
| `m(…)` — a detached reference, e.g. `const f = obj.m; f()` | `undefined` |
| `f.call(r, …)` / `f.apply(r, …)` / `f.bind(r)` | `r` |
| `new C(…)` | the freshly allocated instance |
| arrow function | the enclosing function's `this`, captured lexically |

Two consequences the MVP has to get right. First, the MVP's modules are ESM and
therefore always strict, so the detached case binds `undefined` rather than the
global object, and any property read through it is a reported `TypeError` — the
`const f = obj.m; f()` bug is a defect this frontend should *find*, and modelling
`this` as a fixed leading parameter would silently lose it. Second, an arrow
function has no `this` of its own at all; it is structurally a closure capture,
not a parameter, so it takes the §4.3 path.

Lowering is at conversion time. The annotation pass records the binding form at
each call site; an ordinary function takes `this` as a leading parameter whose
value the *call site* supplies, an arrow function captures `this` into its
closure environment instead of receiving a parameter, and `bind` produces a
closure whose environment holds the bound receiver. Where the receiver is not
statically known, `this` is a tagged value like any other and the `undefined` arm
stays live.

---

## 5. Dynamic typing strategy

### 5.1 The honest starting position

Python's approach is *annotation-directed static monomorphisation with a narrow
tagged escape hatch*. JavaScript has no annotations, so the annotation-directed
half is missing and the escape hatch is too narrow to carry the load (§1.3: a
tagged variable accepts only a numeric-scalar or string rvalue, and JavaScript's
common tagged rvalue is an object, an array or a call result).

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
the C++ converter.

**Scope resolution is work this pass has to do itself.** Acorn parses syntax
into an ESTree tree; it tracks scopes internally only to raise early errors
(duplicate declarations, illegal `await`), and exposes none of that in the AST.
There is no scope tree, no binding table and no reference-to-declaration edge in
Acorn's output. Step 1 below is therefore a real analysis, not a field read —
which also means it is a place the design can be wrong about `var` hoisting or
TDZ and not find out until a differential test fails.

Two ways to get it, and the choice matters for §2.2's zero-dependency claim:

- **Build it in `annotate.js` using `acorn-walk`.** `acorn-walk` ships in the
  Acorn monorepo, is MIT, and has no dependencies of its own, so the vendored
  parser bundle stays dependency-free. Roughly 300 lines: a scope stack over
  `Program`/function/block/`catch`/class nodes, hoisting `var` and function
  declarations to the nearest function scope, `let`/`const`/`class` to the block,
  and binding each `Identifier` in reference position to the nearest enclosing
  declaration.
- **Vendor `eslint-scope`**, the standard ESTree scope analyser. Better tested
  and handles the corners (`with`, `eval`, `arguments`, module bindings) that a
  hand-rolled pass will get wrong, but it pulls `esrecurse`/`estraverse`, so
  "Acorn has zero dependencies" stops describing what is actually vendored.

**Recommendation: `eslint-scope`.** Scope resolution is where subtle unsoundness
enters — a missed TDZ or a mis-hoisted `var` silently changes which definition
reaches a use, and the inference in step 3 is built on top of it. Two extra
vendored files is a smaller cost than a wrong binding table, and the
dependency-free property that matters for §2.4's embedding is *no native
toolchain*, which all three packages satisfy.

With the binding table in hand, the rest is a standard
flow-insensitive-with-SSA-refinement analysis:

1. Bind every identifier to a declaration using the scope tree built above
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

`instanceof` is `isinstance2t`, already resolved by symex against the points-to
set, and `in` is `hasattr2t`. `typeof` is *nearly* a `type_id` read, and the two
places it is not are precisely the two that step 5's narrowing depends on
(ECMA-262, `sec-typeof-operator`):

- **`typeof null === "object"`.** The historical quirk is preserved by the spec
  deliberately, so `null` does not report the distinct tag §4.1 gives it. Worse
  for the analysis, `typeof x === "object"` therefore narrows `x` to
  `object | array | null`, not to `object` — which is why real JavaScript always
  pairs it with `x !== null`, and why the lattice has to narrow away the `null`
  arm on that second test rather than on the first.
- **`typeof f === "function"`.** Closures are `{fn, env}` structs (§4.3) and
  class constructors are shapes, so a raw tag read yields a closure or shape
  tag, never `"function"`. Every callable tag — closure, bound function, class
  constructor, method — has to map onto `"function"`.

`dyn_typeof` is therefore a small total function from tag to result string, not
the identity on `type_id`; it also answers `"undefined"` for an unresolvable
identifier rather than raising, which is the one place `typeof` differs from
evaluating its operand. The mapping is a handful of table rows, so the
*predicates* are still cheap — it is boxing, dispatch and coercion that are new.

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
                     │  ─ dyn_container: ordered map +  │
                     │    indexed seq, presence bits    │
                     │    (from python-list/, -dict/)   │
                     │  ─ dyn_string: string_handler    │
                     │  ─ dyn_object: struct/shape      │
                     │    builder (from class/)         │
                     │  ─ dyn_dispatch: call-site       │
                     │    lowering ONLY — resolution    │
                     │    order supplied per language   │
                     │  ─ dyn_exception: throw/catch/   │
                     │    finally lowering              │
                     │  ─ dyn_scope: symbol_id, module  │
                     │    manager, closure conversion   │
                     └──▲────────────┬─────────────▲────┘
                        │            │             │
              py_resolver (MRO)      │      js_resolver (proto chain)
                                     │
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

#### The interface comes first, and relocation is downstream of it

"Move the Python directories into `src/dynlang/` and add typedefs" is not a
design, it is a `git mv`. Sharing implementation between two frontends without
first agreeing what is shared is how the layer becomes a place where each
language's special cases accumulate behind `if (lang == PYTHON)` — the outcome
§1.5 exists to avoid. So the shared layer is defined as a set of **interfaces
the two converters call**, and code moves only once it sits behind one.

The layer is **seven** interfaces — the same seven the diagram above names and
the §9.1 diagram repeats. Each is stated as *what the consumer may assume*,
because that is the part that has to survive a second consumer.

The third column needs a word of explanation, because an earlier draft of this
table got it wrong in a way worth naming. "Language-specific policy" does not
mean *absent* — every one of these interfaces has per-language behaviour. It
means the behaviour is supplied by the caller **as data**: a coercion table, a
sequence policy, a resolver callback, a method-name map. That is the distinction
M3's entry criterion 3 is actually testing. An interface that branches on the
source language fails; an interface parameterised by a table the frontend hands
it passes, and stays one implementation.

| Interface | Operations | Per-language behaviour, supplied as data |
|---|---|---|
| `dyn_value` | `box(expr,tag)`, `unbox(expr,type)`, `tag_of(expr)`, `type_id_for(name)`, `truthy(expr)`, `binop(op,a,b)`, `equals(a,b,strict)` | the truthiness set and the coercion matrix, as a tag→tag table `truthy`/`binop`/`equals` consume (§5.3) |
| `dyn_container` | ordered map + indexed sequence: `get/set/delete/has`, `len`, `iterate(order)`, presence bits, index-vs-string key partition | a *sequence policy*: whether writing past the end extends `length`, whether shrinking `length` deletes, whether an OOB read raises or yields a sentinel, and which methods skip a hole |
| `dyn_string` | code-unit sequence: `concat`, `slice`, `find`, `compare`, `length` | the method-name map, and locale-sensitive operations. Code-unit **width is a parameter**, not an exclusion — widening it to 16 bits (§4.1, M9) is one change serving both consumers |
| `dyn_object` | shape construction: `struct_typet` from an ordered key list, field read/update, allocation via `__ESBMC_new_object` | how a shape is *inferred*; what a "class" is |
| `dyn_dispatch` | call-site lowering *given* a target set: direct call, guarded candidate set, or fallback; plus devirtualisation bookkeeping | the **resolver callback** that produces the target set — `py_resolver` (MRO) and `js_resolver` (prototype chain, §4.4), each living in its own frontend |
| `dyn_exception` | throw/catch lowering onto `cpp-throw`/`cpp-catch`, `finally` duplication onto every exit edge (§4.5) | the built-in error hierarchy, and what may be thrown |
| `dyn_scope` | `symbol_id` naming, module resolution and merging, closure-environment materialisation | the module resolution algorithm; the name-mangling prefix (`py:`/`js:`) |

Three of those rows are corrections to how this table read before, and they are
worth stating rather than quietly fixing:

- **`dyn_container` owns the presence bit and the `props` field, not just an
  abstract map.** §4.2.1 puts `present[]` and `props` in the shared layout on
  purpose: they are *storage*, and storage is exactly what belongs in a shared
  container. What is per-language is the *policy* over that storage — whether
  `forEach` skips a hole, whether shrinking `length` deletes above it, whether
  an OOB read raises. Python constructs its sequences with a no-coupling,
  OOB-raises policy; JavaScript with a coupling, OOB-yields-`undefined` one.
  Same code, different table.
- **`dyn_value` carries `binop` and `equals`.** §5.3 says they must be built,
  so leaving them off the interface was an omission, not a scoping decision.
  They are also where the coercion matrix is consumed, which is why the third
  column names the matrix as data rather than pretending coercion lives
  somewhere else.
- **`dyn_dispatch` is an interface with per-language implementations behind
  it** — which is what §9.3 means by labelling that row **interface** rather
  than **shared**, and the label is correct rather than anomalous. The shared
  half is real and non-empty: call-site lowering, the guarded-candidate `if2t`
  chain, and the devirtualisation bookkeeping are one implementation. The
  resolver half is a callback, and neither `py_resolver` nor `js_resolver`
  lives in `src/dynlang/`.

`dyn_dispatch` is the row that decides whether this works. Python resolves a
method by C3 linearisation over `__mro__`; JavaScript walks a `[[Prototype]]`
chain that is mutable at runtime (§4.4 step 2 states exactly when it can be
resolved statically and what has to be proved first); C++ — whose
`clang_cpp_adjust` the Python frontend currently borrows for exactly this — uses
static vtable offsets fixed at compile time. These are three different
algorithms, and they are the one thing in this layer that parameterisation
genuinely cannot reconcile — which is why the resolver is a callback rather than
a table like the other six rows.

This is also the correction to §4.2's claim that ES6 classes map onto
`python_class_builder` "with almost no change". The *struct layout* and the
allocation path do carry over. The resolution does not: `python_class_builder`
reaches `clang_cpp_adjust` and gets C++ base-subobject arithmetic, which is
right for Python's static-ish MRO and wrong for a prototype chain that can be
reassigned. §4.4's ladder is the JS resolver, and it lives in
`src/js-frontend/js_object/`, not in `src/dynlang/`.

#### Then, and only then, relocation

Once an interface exists and the Python frontend has been rebuilt against it,
the implementation moves: `src/python-frontend/{python-list,python-dict,string,
set,tuple,class,exception}/` and `src/c2goto/library/python/*.c` relocate to
`src/dynlang/` and `src/c2goto/library/dyn/`, with `PyObject`/`PyListObject`
retained as `typedef` aliases so the ~4.6k tests under `regression/python/` keep
passing unchanged. The Python frontend keeps what is genuinely Python: PEP 484
annotation handling, the `models/*.py` library, numpy, complex numbers, MRO
resolution, and the Python-specific converter dispatch.

A component that cannot be stated as one of the seven interfaces without a
language conditional does not move — where "conditional" means a branch on the
source language, not a table the frontend supplies. It is duplicated on the JS
side and both copies are maintained — the cost §1.5 accepts on purpose.

This is the answer to the final requirement in the brief: not a new dynamic
representation, and not a shared directory either, but a shared *interface* over
the existing representation, with a second consumer to prove the interface is
real.

---

## 6. Runtime modelling

### 6.1 Reused from the Python infrastructure

| Area | Source | Notes |
|---|---|---|
| String storage and methods | `string/` (9.5k lines), `library/python/string.c` | large method overlap; name mapping |
| Array/list storage | `python-list/` (8.3k lines), `library/python/list.c` | infinite-array model carries over |
| Keyed containers | `python-dict/` (4.8k lines) | backs open objects and `Map` |
| Sets | `set/python_set.cpp` | backs `Set` |
| Object allocation | `__ESBMC_new_object` + its symex handler (`symex_main.cpp:1058-1116`) | one typed, non-expiring dynamic object per site; reference semantics |
| Infinite containers | `__ESBMC_create_inf_obj` → `symex_mem_inf` | the unbounded `items` array behind symbolic indices |
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
| `this` binding (§4.6) | conversion-time: method call, detached call → `undefined`, `call`/`apply`/`bind`, arrow lexical capture | 6 |
| Coercion (`==`, `+`, `ToPrimitive`) | explicit lowering table; `--js-strict-equality` warns on loose `==` | 4 |
| `ToInt32` bitwise semantics | conversion helper (§4.1) | 4 |
| Array holes / OOB → `undefined` | shared array OM, driven by the JS *sequence policy* (§5.5), not a language branch | 5 |
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
  zero, arithmetic overflow on the integral path, uncaught exceptions,
  null/undefined property access, type confusion on unbox. **Array bounds are
  opt-in** (`--js-array-oob=error`, §4.2.1): JavaScript defines an out-of-range
  read as `undefined` rather than as an error, so reporting it is a
  stricter-than-the-language check a user asks for, not a default.

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
| Property checks | `goto_check` — division by zero, overflow, pointer safety, and bounds on the *model's own* buffers, all for free. JS index bounds are **not** among them: an out-of-range read is in bounds of the infinite `items` array and is a JS-level property the array OM raises under `--js-array-oob=error` (§4.2.1) |
| Symbolic execution | `goto-symex`, incl. `isinstance2t` resolution, the `__ESBMC_new_object` handler and `symex_mem_inf` (the infinite-array allocator behind `__ESBMC_create_inf_obj`) |
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
 │  libs/acorn/                    vendored parser (MIT)        │
 │  libs/acorn-walk/               traversal (MIT)              │
 │  libs/eslint-scope/             scope analysis (BSD-2, §5.2) │
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
 │   dyn_dispatch/ (lowering only — resolver supplied by the    │
 │                  frontend, §5.5)  dyn_exception/  dyn_scope/ │
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
| shapes, struct layout | `class/python_class_builder.cpp` | **shared** (`dyn_object`) |
| method/property resolution | `dyn_dispatch` call-site lowering only | **interface** — the resolver is a per-language callback; the JS one is new (§4.4, §5.5) |
| exceptions | `exception/` + `remove_exceptions` | **shared** |
| modules | `module/module_manager.cpp` | **shared** |
| naming | `symbol_id` | **shared** |
| tags | `PyObject` → `DynObject` | **shared** |
| `typeof`/`instanceof`/`in` | `isinstance2t`/`hasattr2t`/`isnone2t` + symex | **shared** |

### 9.4 New JavaScript-specific components

Eight: the Acorn integration, scope resolution (§5.2 — Acorn supplies no scope
tree), the type-inference pass, object-shape inference, **prototype resolution
as its own resolver** (§5.5 — not shared, because MRO and prototype chains are
different algorithms), closure conversion, **`this` binding** (§4.6 — determined
by the call site, so it is not Python's `self` under another name), and the
coercion tables (`ToInt32`, `ToPrimitive`, `ToBoolean`, `==`).

Plus two generalisations that are new *work* even though they land in shared
code: presence bits and index/string key partitioning in `dyn_container`, and
the `length` invariant in the array OM (§4.2.0). Earlier drafts of this document
counted five, described the array change as a field rename, and treated `this`
as pure reuse; all three were wrong. The count is stated here so the estimate in
§10 is not read as smaller than it is.

---

## 10. Roadmap

Each milestone is independently mergeable, keeps `regression/python/` green, and
ships at least one passing and one failing regression test per the project's PR
convention.

### M1 — Acorn integration and language registration

**Deliverables.** `js_languaget` registered for `.js`/`.mjs`/`.cjs`; Acorn,
`acorn-walk` and `eslint-scope` vendored under `src/js-frontend/libs/` and
FLAIL-mangled; `parse.js` driver; `--parse-tree-only` dumps ESTree JSON; Node
discovery and version check.
**Risks.** Node on `PATH` (accepted; M7 removes it). FLAIL asset layout for a
multi-file JS bundle.
**Testing.** Round-trip a corpus of ~50 JS files through `--parse-tree-only` and
compare against `node -e "acorn.parse(...)"` under the structural comparison of
§2.4.
**Validation.** Every file parses; every tree matches a direct Acorn run
structurally (byte equality is not required, and not asserted — see §2.4).

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

### M3 — Shared dynamic-language layer

**Gated, not scheduled (§1.5), and split in two, because the two halves have
different preconditions and only one of them can run early.**

#### M3a — interfaces, and Python rebuilt against them in place

Entry criteria; all three must hold before a line of `src/dynlang/` is written:

1. The seven `src/dynlang/` interfaces (§5.5) are written down as headers, with
   each operation's contract stated, and reviewed by a maintainer of the Python
   frontend.
2. The Python frontend has been rebuilt against those headers *in place* —
   still under `src/python-frontend/`, no files moved — and `regression/python`
   is green. This is the step that proves the interface is adequate before it
   is load-bearing for two languages.
3. No interface operation branches on the source language. Per-language
   behaviour is supplied as data — a coercion table, a sequence policy, a
   resolver callback (§5.5) — and a component that needs a branch instead is
   struck off the shared list and duplicated.

M3a involves no JavaScript at all. That is what makes it the part that can be
worked in parallel with M1 and M2.

#### M3b — relocation

Entry criterion: **a JavaScript consumer actually calls the interfaces**, so
they have a second consumer rather than a hypothetical one.

That consumer does not exist at M2. M2's deliverables are deliberately
monomorphic — numeric literals, arithmetic, control flow, intrinsics — and touch
neither `dyn_value` nor `dyn_container`. The first real traffic is M4's widened
`dyn_assign`/`dyn_binop`/`dyn_unbox` and M5's arrays and objects. **M3b is
therefore sequenced after M5**, and until it lands the JS frontend calls the
interfaces where M3a has defined them and duplicates where it has not.

If M3a's criteria are never met, M3b never happens: the JS frontend proceeds
entirely on duplicated implementations and the rest of the roadmap is unaffected.
That is the point of §1.5 constraint 3 — this plan has to survive its own
largest refactor not happening.

**Deliverables.** *M3a:* interface headers; the Python frontend rebuilt against
them in place; `models/errors.js`, `models/nondet.js`. *M3b:* `src/dynlang/` and
`src/c2goto/library/dyn/` created by moving Python components behind the agreed
interfaces; `PyObject`/`PyListObject` retained as typedefs; per-language
resolvers (`py_resolver`, `js_resolver`) left in their own frontends.
**Risks.** *This is the highest-risk milestone* — it touches code covered by
~4.6k tests, and it freezes interfaces on a frontend that is still moving.
Mitigation: the entry criteria above; then pure relocation, no behaviour change,
one component per commit, full `regression/python` after each.
**Testing.** `regression/python` must be verdict-identical before and after
*each* half; capture verdicts to a baseline file and diff. An include-path lint
asserting `src/js-frontend/` never includes `src/python-frontend/` and vice versa
(§1.5 constraint 2), wired into CI.
**Validation.** Zero Python regressions; the include lint passes; the JS
frontend links only against `src/dynlang/`, never `src/python-frontend/`.

### M4 — Type inference, coercion, and the tagged path

**Deliverables.** `annotate.js` (scope resolution via vendored `eslint-scope`,
lattice, fixpoint, narrowing);
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

**Deliverables.** Array literals, indexing, `length` as a maintained invariant,
mutation and query methods over the shared array OM; **presence bits and hole
semantics**; **index-vs-string key canonicalisation** and the enumeration order
of §4.2.0 #5; OOB→`undefined`; object literals with shape inference →
`struct_typet`; open objects → ordered dict; non-index properties on arrays;
destructuring and spread desugaring; `Object.keys/values/entries`.
**Risks.** Shape inference misjudging an object as closed when a later branch
adds a key, or when creation order differs between paths — must be conservative
in both, and the desugaring must be verified against Node. Hole semantics are
per-method and inconsistent in the language itself (`forEach` skips, `map`
preserves, spread fills), so the method table has to encode the split rather
than pick one rule.
**Testing.** ~120 regression tests, including a sparse-array set that
distinguishes `[1,,3]` from `[1,undefined,3]` under `in`, `Object.keys`,
`forEach`, `map` and spread; an enumeration-order set mixing integer-like and
string keys; property-based comparison against Node for array method semantics.
**Validation.** Bounds properties fire correctly; a shaped object costs the same
as the equivalent C struct in VCC count; zero differential mismatches against
Node on the sparse-array and key-order sets.

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
backends produce structurally identical ASTs on the whole corpus, compared after
canonical serialisation (§2.4) rather than on raw engine output.
**Validation.** ESBMC verifies a multi-module JS program with no `node` on
`PATH`; both backends agree on every tree in the corpus under the structural
comparison.

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
riskiest. M1, M2 and **M3a** can be worked in parallel by separate contributors
once M1's JSON contract is fixed — M3a is a Python-side refactor and depends on
no JavaScript code. M5 and M6 both depend on M4's annotations, and **M3b depends
on M5**, because M5 is the first point at which a JavaScript consumer exercises
the interfaces M3b relocates.

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
   plus 4.2k lines of C operational models, plus ~4.6k regression tests. A
   parallel JS-only representation would mean reimplementing all of it *and*
   maintaining two divergent models of the same abstractions.

3. **The seams are already language-neutral.** `isinstance2t`/`hasattr2t`/
   `isnone2t` are irep2 nodes with symex lowerings that query the points-to
   set — they know nothing about Python. `__ESBMC_new_object`'s symex
   interception, the infinite-array container trick, the `cpp-throw` reuse and
   `clang_cpp_adjust` are all generic mechanisms that happen to have been
   reached through Python first.

4. **The one real gap is a gap in the Python frontend too.** Assigning a
   non-scalar, non-string rvalue to a tagged variable throws today
   (`converter_stmt.cpp:3059-3076`). Fixing that behind `dyn_value` fixes both
   languages; forking it fixes one and leaves the other with a
   `runtime_error`. The same argument applies to bignum, to UTF-16, and to
   `try`/`finally` with escaping control flow — three more places where the JS
   requirement is the forcing function for a Python fix.

The honest counterweight, restated from §1.5: (2) and (3) are arguments for
*reusing* the existing mechanisms, not for *merging the two frontends' code*.
The proposal takes reuse and declines the merge — §5.5's interface boundary and
§10's M3 entry criteria are what keep those two things separate, and if the
criteria cannot be met, the reuse happens by duplication and the conclusion
above is unaffected.

Where JavaScript genuinely differs from Python — prototype chains, capturing
closures, `this` bound at the call site, the `ToInt32` bitwise rules, `undefined`
as distinct from `null`, `==` coercion — the proposal adds *JavaScript-specific
components on top of the shared layer*, not a competing substrate. Those
differences are what §9.4's eight new components exist to cover: the irreducible
JavaScript-specific surface, and nothing more.
