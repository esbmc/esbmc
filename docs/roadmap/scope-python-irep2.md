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

## 4. Next

- The 22 type writes §3 identifies as blocked, once `#cpp_type` has a route.
- The 48 value writes, which need §52's namespace precondition and nothing else so
  far as this census can tell.
- `#cpp_type`: census its readers the way `scope-solidity-irep2.md` §9 censused
  Solidity's, and pick a route from the four in that document's §11.1.
