# Phase 9 (python): scope

Phase 9 is the python frontend's share of the IREP2 migration, as numbered in
`frontends-to-irep2.md` §39.3. This document is the scope note; measurements are
dated so a stale one is visible as stale.

## 1. B-2 censused (2026-09-15)

`git grep 'set_type(\|set_value(' -- src/python-frontend | grep -vc 2tc` is **108**:
**60** type writes and **48** value writes, spread over 26 files, with
`converter/converter_stmt.cpp` (17), `converter/converter_funcdef.cpp` (13) and
`python_converter.cpp` (11) holding a third.

### 1.1 The attribute surface is four, not eleven

The comparison that matters is with Solidity, whose type writes are blocked by eleven
Solidity-level attributes on legacy `typet` nodes (`scope-solidity-irep2.md` §7).
Every `#`-prefixed attribute the python frontend mentions:

```
#member_name   #location   #identifier   #cpp_type
```

`#member_name` lost its last reader when the C++ frontend stopped reading it
(`frontends-to-irep2.md` §50), and `#location` / `#identifier` are carried by the seam
already. So python's exposure is one attribute: `#cpp_type`.

## 2. What converts, measured (2026-09-15)

Converting every single-line type write -- 27 of the 60; the rest span lines and were
left for a later pass -- fails **3 of the first 400** python tests:
`class-attributes-scoped`, `collections_annotation`,
`complex_math_typeerror_edges`, each with a spurious `TypeError`.

Bisecting names one site for all three: `python_converter::create_symbol`
(`converter/converter_util.cpp:32`), the shared factory behind 38 call sites. With it
left alone and the other 23 converted, the suite is **400 of 400 and 500 of 500** over
the two slices the ten-minute cap allows, and the unit suite is 876 of 876.

### 2.1 Why that one site is blocked

`#cpp_type` is the source-level spelling of a type, and `migrate_type` drops it. The
frontend already knows: `python_expr_builder.cpp` carries five comments saying so, and
restores the exact type after migrating around it. `create_symbol` takes a `typet` and
stores it for 38 callers, so converting it drops the spelling for all of them and the
python type checker reports a `TypeError` that is not in the program.

That makes `#cpp_type` python's analogue of `#sol_type`, with two differences that
matter: it is one attribute rather than eleven, and its readers are presentation
consumers plus the type checker rather than the whole frontend
(`util/irep/irep.h:493` describes the three presentation ones). Whether it can be
derived, side-tabled or read from the AST is the question Phase 9 has to answer, and
it is a much smaller question than Phase 8's.

## 3. The arrow-form writes: 9 of 33 (2026-09-15)

The 33 writes §2 left were not multi-line -- they spell `sym->set_type(...)`, which the
first pass's pattern did not match. Converting them measures as follows.

**Two of the 33 were false positives of the B-2 grep itself**: `python_adjust.cpp:65`
and `:97` pass a `type2tc`, so they already wrote IREP2 and only matched because the
grep counts the argument's spelling. That is the bar's known property (§39 of the parent
doc) showing up in the tooling used to survey it.

Of the remaining 31, **9 land clean** -- 400 of 400 and 500 of 500 over the two slices,
unit 876 of 876. Three sets do not:

| site | symptom | cause |
|---|---|---|
| `converter/converter_stmt.cpp` (11 writes) | `casting14` fails | most write `rhs.type()`, which carries `#cpp_type` |
| `converter/converter_funcdef.cpp` (10 writes) | `class_var_param_augassign{,_fail}` fail | same shape, parameter and return types |
| `python_adjust.cpp:70` | the unit case `python_adjust pre-pass write-back preserves bases for the throw chain` fails | the write exists to re-attach the legacy-only `bases` sub-irep, and storing IREP2 drops it again |

The last is worth its own line: that write's own comment says what it is for, and the
repo already had a unit test pinning it. A scripted conversion walked into it and the
test caught it immediately -- which is the argument for the test, not against the
script.

### 3.1 Two attributes, not one

§2.1 called `#cpp_type` python's single exposure. The `bases` case adds a second, and it
differs in kind: it has no `#`, so it lives in `named_sub` and takes part in `typet`
equality, and it is a *struct* attribute rather than a scalar's spelling. Its consumers
are named in the code -- `derive_exception_ids`, `exception_typeid.cpp`,
`base_type.cpp`.

So Phase 9's blocked set is `#cpp_type` (the type checker plus three presentation
consumers) and `bases` (the exception hierarchy). Still a far smaller surface than
Phase 8's eleven, and both are single questions rather than families.

## 4. The value writes: 7 of 48, and a bigger wall than the types (2026-09-15)

§2.1 read the census as saying the value writes need only §52's namespace precondition.
Two measurements say otherwise.

### 4.1 The precondition is already satisfied, and is not the problem

`migrate_namespace_lookup` is never pointed at the python frontend's context, so the
first question was whether it should be. Instrumenting `sym_name_to_symbol` over 60
tests: **750 449 hits and 177 472 misses**, a 19% miss rate. Pointing it at
`python_converter::ns` changes the miss count by **zero** -- 60 160 before and after on
a 20-test sample -- because that namespace and the one `language_ui` installed wrap the
same context. Unlike the C++ adjust pass (§52), python's precondition already holds.

The misses are genuine: `python_converter::<file>:N$list_size$N` and friends are
internal temporaries never added to the table, and a smaller group are model-function
parameters looked up before they are added. Python ids carry no `#` or `&`, so the
renaming parser's mangling (§55) cannot bite here; a miss costs the symbol-table type,
not the name.

### 4.2 What converts

`migrate_expr` had only an out-parameter form, which is why a value write is two
statements. A returning overload in `util/irep/migrate.h` makes each one a one-liner,
and is what the remaining conversions in every frontend will want.

With it, of 48 value writes: six take a string literal and are not expressions at all,
one in `python_adjust.cpp` already passed an `expr2tc` (another grep false positive),
and of the remainder **7 convert cleanly** -- 400 of 400, 500 of 500, unit 876 of 876.
The rest fail hard, and the failure is not subtle: converting
`converter/converter_stmt.cpp`, `converter_funcdef.cpp`, `converter_symbols.cpp`,
`python_converter.cpp` or `python2goto.cpp` takes the first slice to **389 failures of
400**.

That is a different kind of wall from the type writes. A type write drops an attribute
and one consumer notices; these writes are the ones that build the program's bodies and
`main`, and storing them IREP2-side at conversion time breaks nearly everything -- which
is what §49.1 measured for C++ before §52 explained it, except here the namespace
explanation is ruled out by §4.1. The cause is not yet known, and finding it is the next
task rather than a guess to record.

## 6. The wall diagnosed: a body cannot be migrated before its symbols exist (2026-09-15)

§4.2 left the hard failure unexplained. Narrowing it one site at a time gives the
answer, and it is not the 34 sites the earlier batch implicated.

Eight more value writes convert cleanly -- all three in
`converter/converter_symbols.cpp`, both in `python2goto.cpp`, and three of
`python_converter.cpp`'s five. The two that do not are the **program-entry bodies**:

```cpp
user_main_symbol.set_value(user_code);   // python_converter.cpp:1057
main_symbol.set_value(std::move(v));     // :1179
```

Converting `user_main` alone takes the first slice to **53 failures of 400**, and the
failure is a SIGSEGV during GOTO creation rather than a wrong verdict.

### 6.1 Why, and the rule it gives

§4.1 measured a 19% symbol-lookup miss rate in this frontend, and called the misses
harmless because python ids cannot be mangled. They are harmless *while values stay
legacy*: a legacy value is migrated later, by which time the symbol table is complete
and every lookup hits. Migrating a body **at conversion time** freezes those misses
into the stored expression -- each missed symbol keeps the expression's own type
instead of the symbol-table type, which `migrate.cpp` warns hashes wrongly -- and a
body references every symbol the module declares, so the 19% is spread across the whole
program.

`user_code` is the entire user program. That is why a small initialiser converts and a
body does not, and it is the same shape as `frontends-to-irep2.md` §49.1's C++ finding
with a different cause: there the namespace was wrong, here the namespace is right
(§4.1) and the symbols genuinely are not there yet.

So the rule for the rest of Phase 9, and for any frontend: **a symbol's value may be
stored IREP2-side at conversion time only if every symbol it names is already in the
table.** For bodies that is false by construction, and the laziness of
`set_value(const exprt &)` is what makes the legacy path correct -- migration happens
after the table is complete. Moving a body to IREP2 therefore belongs in the adjust
pass, which runs after the link, not in the converter.

## 7. Next

- The 22 type writes §3 identifies as blocked, once `#cpp_type` has a route.
- The two entry-body writes, in the adjust pass rather than the converter (§6.1).
- `#cpp_type`: census its readers the way `scope-solidity-irep2.md` §9 censused
  Solidity's, and pick a route from the four in that document's §11.1.

## 8. `#cpp_type` censused, and `irep.h` was wrong about its readers (2026-09-15)

§7 asked for this census. The attribute has **five writers** and, across `src/` and
`unit/`, these readers:

| reader | kind |
|---|---|
| `util/lang/cpp_expr2string.cpp:138,140` | presentation (counterexample text) |
| `goto2c/expr2c.cpp:174` | presentation (generated C) |
| `clang-cpp-frontend/clang_cpp_exception_id.cpp:45` | exception-id strings |
| `python-frontend/type/type_utils.h:207` | **verifier core** |

`irep.h`'s own comment said "its three readers are all presentation consumers ...
rather than verifier core". That is the first three. The fourth is
`type_utils::is_char_type`:

```cpp
static bool is_char_type(const typet &t)
{
  return (t.is_signedbv() || t.is_unsignedbv()) && get_cpp_type(t) == "char";
}
```

-- whether an 8-bit bitvector is a Python character or an `int8` -- and **seven
conversion sites branch on it**, in `converter_binop.cpp` (three),
`string_handler.cpp` (two), `tuple_handler.cpp` and `str_conv.cpp`. Dropping the
spelling changes what is verified, not how it is printed. The comment is corrected in
this change; a comment naming a reader set that is not the reader set is the trap
`frontends-to-irep2.md` §56 removed four stores to avoid.

### 8.1 Which route fits

`is_char_type`'s question is a collision of exactly the shape
`scope-solidity-irep2.md` §13.1 found for `ADDRESS` against `UINT160`: two Solidity --
here Python -- kinds over one IREP2 type. So the derivation route (§8 of that document)
does not apply.

The side table (§10 there) does not apply either as written: `is_char_type` is asked of
a subtype (`arr_type.subtype()` at `string_handler.cpp:2397`), where there is no symbol
to key on.

That leaves the AST route (§11 there) and one option those documents do not list,
available here because the question is boolean and the domain is tiny: **give IREP2 the
distinction**. A python character is not a spelling detail the way `long long` is -- it
is a different type in the source language, and `unsignedbv` of width 8 is the wrong
model for it. That is a type-system question for the python frontend rather than a
migration one, and it is the last thing standing between Phase 9 and its 22 blocked
type writes.

### 8.2 What is left, and what is not a gap

Two of Phase 9's remaining writes should **stay legacy**, and counting them as debt is a
mistake B-2's spelling-based count invites. The program-entry body writes (§6) are
correct as they are: `set_value(const exprt &)` defers migration until the symbol table
is complete, which is exactly what a body needs. Converting them is not blocked work, it
is work that must not be done.
