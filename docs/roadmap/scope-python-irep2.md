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

### 2.0 A second blocked site, and what the slices could not see

The full suite then failed `lambda_default_arg` on the llvm-22 job. The cause is the
same attribute at a second site: `python_lambda::create_symbol`
(`lambda/python_lambda.cpp:238`) shares only a name with the factory above, and was
converted because the two slices never reached it -- the test is #12255, and the slices
stop at 500. Without `#cpp_type`, a `bool` default lowers as `double`, so
`lambda x, flag=True: flag` yields `return_value$ == (double)1`.

So the batch is **22**, not 23, and the sampling is the lesson: a slice bounded by the
ten-minute cap is a smoke test, not a bisect. A site the slices do not reach cannot be
called clean on their evidence.

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

## 3. The arrow-form writes: 6 of 33 (2026-09-15)

The 33 writes §2 left were not multi-line -- they spell `sym->set_type(...)`, which the
first pass's pattern did not match. Converting them measures as follows.

**Two of the 33 were false positives of the B-2 grep itself**: `python_adjust.cpp:65`
and `:97` pass a `type2tc`, so they already wrote IREP2 and only matched because the
grep counts the argument's spelling. That is the bar's known property (§39 of the parent
doc) showing up in the tooling used to survey it.

Of the remaining 31, **6 land clean** -- unit 876 of 876. Five sets do not:

| site | symptom | cause |
|---|---|---|
| `converter/converter_stmt.cpp` (11 writes) | `casting14` fails | most write `rhs.type()`, which carries `#cpp_type` |
| `converter/converter_funcdef.cpp` (10 writes) | `class_var_param_augassign{,_fail}` fail | same shape, parameter and return types |
| `python_adjust.cpp:70` | the unit case `python_adjust pre-pass write-back preserves bases for the throw chain` fails | the write exists to re-attach the legacy-only `bases` sub-irep, and storing IREP2 drops it again |
| `class/python_class_builder.cpp` (the two `set_type(st)` writes) | 13 exception tests, `is-instance`, `shedskin` and `mopsa/try_super_raise` fail | `get_bases(st)` has just attached `bases` to the struct type; this is `python_adjust.cpp:70`'s cause at the site that produces it rather than the one that repairs it |
| `lambda/python_lambda.cpp:54` | `lambda_default_arg` fails | `#cpp_type` again, on the function-pointer type a lambda binding takes |

The two added rows are the reason the slice numbers are gone from this section. The
first pass reported "400 of 400 and 500 of 500", and every one of these 15 tests sits
past 500 -- §2.0's lesson, arrived at twice. What the section can claim is the unit
suite and the named tests, so that is what it claims.

The `python_adjust.cpp:70` row is worth its own line: that write's own comment says what
it is for, and the repo already had a unit test pinning it. A scripted conversion walked into it and the
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

## 5. Next

- The 22 type writes §3 identifies as blocked, once `#cpp_type` has a route.
- Why the converter's body and `main` value writes fail as hard as §4.2 measures --
  34 sites, and the namespace explanation is already ruled out.
- `#cpp_type`: census its readers the way `scope-solidity-irep2.md` §9 censused
  Solidity's, and pick a route from the four in that document's §11.1.
