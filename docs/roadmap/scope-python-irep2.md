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

## 6. The wall diagnosed: a body cannot be migrated before its symbols exist (2026-09-15)

§4.2 left the hard failure unexplained. Narrowing it one site at a time gives the
answer, and it is not the 34 sites the earlier batch implicated.

Seven more value writes convert cleanly -- two of the three in
`converter/converter_symbols.cpp`, both in `python2goto.cpp`, and three of
`python_converter.cpp`'s five. The third `converter_symbols.cpp` write has its own
cause, recorded in §6.2. The two that do not are the **program-entry bodies**:

```cpp
user_main_symbol.set_value(user_code);   // python_converter.cpp:1057
main_symbol.set_value(std::move(v));     // :1179
```

Converting `user_main` alone takes the first slice to **53 failures of 400**, and the
failure is a SIGSEGV during GOTO creation rather than a wrong verdict.

### 6.2 A retyped value cannot be migrated either (2026-09-16)

`update_symbol`'s first write is the third site that has to stay legacy, and the reason
is not §6.1's:

```cpp
const typet &expr_type = expr.type();
sym->set_type(migrate_type(expr_type));
exprt v = sym->get_value();
v.type() = expr_type;          // retypes the root only
sym->set_value(v);             // must stay legacy
```

The assignment retypes the value's **root node** and leaves its operands alone. A legacy
`exprt` tolerates that; IREP2 does not. Migrating eagerly builds an arithmetic node whose
result type is `expr_type` while operand 1 keeps the type it had, and
`assert_arith_2ops_consistency` (`irep2_expr.cpp:698`) rejects it:

```
Assertion `p2 || (is_bv_type(t) == is_bv_type(v1->type)
                  && t->get_width() == v1->type->get_width())' failed.
```

`regression/numpy/div1` reaches it: `np.divide(1, 2)` has no inferable return type, the
frontend defaults it to `double` (`converter_funcdef.cpp:1933`), and the retyped root
then sits over integer operands. Six numpy division tests abort this way, and because it
is an assertion it is invisible in any build with `NDEBUG` -- it surfaced on the llvm-22
DebugOpt job, not in the 400/500 slices.

The rule this adds to §6.1's: a value write can be converted only if the expression is
*already* consistent. Retyping the root is a legacy idiom that the storage flip turns
into a hard error, so a site that does it needs the retype pushed through the operands
before the write can move -- which is a change to what the frontend builds, not to where
it stores it.

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

## 9. §3's prediction, confirmed — and the cost of not reading §3 (2026-09-17)

`handle_assignment_type_adjustments` (`converter_stmt.cpp:1200-1455`) holds twelve B-2 writes, the
densest single-function cluster left in the migration. Eleven were converted, measured, reviewed, and
**reverted**: they break seven CORE tests. §3 of this document already said they would.

### 9.1 The finding, and the two-line reproducer

```python
val = "hello"[0]
assert val == "h"
```

```
base arm                VERIFICATION SUCCESSFUL
eleven writes converted VERIFICATION FAILED     (GOTO shows ASSERT 0)
```

Attributed by stashing the change and rebuilding, not inferred. Seven CORE tests fail the same way and
pass on base: `casting14`, `enumerate8`, `for-loop3`, `for-loop6`, `for-loop8_fail`,
`python_irep2_adjust_only_string_index`, `string-concat6`. `assert "hello"[0] == "h"` is unaffected --
the defect needs the char to pass *through a symbol*, which is exactly what these writes change.

The chain, each link at a line:

```
list_access.cpp:4119      tags a string-subscript result #cpp_type == "char"
converter_stmt.cpp:1402   set_type(migrate_type(rhs.type())) on that tagged type
migrate.cpp:3238          rebuilds signedbv_typet(width); #cpp_type is not carried
converter_expr.cpp:1763   a Name read is typed from symbol->get_type()
converter_binop.cpp:619   get_python_type_category returns "numeric", not "string"
                          -> the comparison folds cross-type to false
```

`list_access.cpp:4119`'s own comment names the consumer it is feeding, which is the one that breaks.

### 9.2 §3 predicted exactly this, in this file

§3's table, above:

| site | symptom | cause |
|---|---|---|
| `converter/converter_stmt.cpp` (11 writes) | `casting14` fails | most write `rhs.type()`, which carries `#cpp_type` |

Eleven writes, this file, `casting14`, `#cpp_type`. An earlier pass had already measured it. The work
above re-measured it from scratch, reached the opposite conclusion, and appended a section to the same
document without reading the section three headings up. That is the whole finding worth keeping: the
answer was in the file being edited.

### 9.3 Why three instruments all missed it

None of them was aimed at the attribute.

- **The A/B is structurally blind to `#cpp_type`.** `python_languaget::from_type` routes to
  `c_type2string` (`python_language.cpp:362-370`), and `c_expr2string.cpp` never reads `#cpp_type` --
  only `cpp_expr2string.cpp:138` and `goto2c/expr2c.cpp:174` do. The failing case prints
  `Type........: signed char` either way, because the width-8 fallback produces that string without the
  tag. So "0 of 76 differ" was evidence about branch flips in 76 programs, not about the attribute, and
  reading it as corroboration was wrong.
- **The census sampled the shape out.** It counted `#cpp_type` on writes in the 78-test stratified
  corpus: 51 `double`, 14 `unsigned_long`, no `char`. But site `:1402` runs in 338 tests and the corpus
  held 78, chosen for *site execution* rather than for the attribute -- so the `char`-spelled writes
  were simply not in it. §9.1 of the reverted text criticised precisely this error about an earlier
  filter and then repeated it one section later.
- **The census also read the wrong node.** `is_char_type` is asked of an array *subtype*
  (`string_handler.cpp:2397`, `:2431`), and a probe on `rhs.type().cpp_type()` cannot see a tag on
  `rhs.type().subtype()`.

A census aimed at an attribute must enumerate its **writers** (`type_handler.cpp:651`,
`list_access.cpp:4120`, `convert_float_literal.cpp:26/31/36`) and route each through each site. Counting
what a convenience corpus happens to contain measures the corpus.

### 9.4 What this changes in the plan

These eleven writes are blocked on the `#cpp_type` carry §8.1 scopes -- not blocked on more measurement.
So is the next cluster: §3's table also lists `converter_funcdef.cpp`'s ten writes failing
`class_var_param_augassign{,_fail}` for the same reason, and `python_adjust.cpp:70` for the analogous
loss of the legacy-only `bases` sub-irep.

The frontend already compensates for this loss by hand at six call sites in
`python_expr_builder.cpp` (`:40`, `:71`, `:90`, `:112`, `:176`, `:312`), each commented "migrate_type
does not round-trip `#cpp_type`; restore the exact target type". Seven ad-hoc restorations is the
argument for the carry, not for an eighth.

So Python's B-2 residue stays 54, and the next task is the carry itself, with a regression pair over
`val = "hello"[0]; assert val == "h"` added in the same change so a later attempt at these eleven cannot
pass review silently.
