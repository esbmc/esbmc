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
## 4. Next

- The 22 type writes §3 identifies as blocked, once `#cpp_type` has a route.
- The 48 value writes, which need §52's namespace precondition and nothing else so
  far as this census can tell.
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

So the next task is the carry itself, with a regression pair over
`val = "hello"[0]; assert val == "h"` added in the same change so a later attempt at these eleven cannot
pass review silently.

## 10. The carry, and the eleven writes with it (2026-09-17)

§9 left the cluster blocked on `#cpp_type`. The attribute now crosses the seam, and the eleven writes
land behind it.

### 10.1 Three kinds, chosen by writer

`irep_idt cpp_type` is added to `unsignedbv_type2t`, `signedbv_type2t` and `floatbv_type2t`, carried
both ways in `migrate_type`/`migrate_type_back`, with the back-write guarded on non-empty -- `#cpp_type`
is a comment field, and writing it empty inserts a key the printer reads, which is the §158 hazard for
the fourth time.

§9.3's mistake was censusing a corpus, so the kinds come from enumerating the **writers** --
`type_handler.cpp:651` and `list_access.cpp:4120` both tag `char_type()`, i.e. signedbv, and
`convert_float_literal.cpp:26/31/36` tag `float_type()`, `long_double_type()` and `double_type()`, i.e.
floatbv. Two qualifications the first draft of this section got wrong:

- **unsignedbv is not carried for a Python writer.** `char_type()` is unsignedbv when
  `config.ansi_c.char_is_unsigned` (`c_types.cpp:190`), so the arm exists for platform symmetry, and it
  is the arm `goto2c/expr2c.cpp:174` reads. The `unsigned_long` spellings §9.3 counted come from
  `clang_c_convert.cpp:1549`, in the **C** frontend; nothing in `src/python-frontend/` writes that
  spelling.
- **The enumeration is of Python's writers, not the tree's.** `clang_c_convert.cpp:1503-1607` tags
  every clang builtin, and around fifteen Solidity sites tag theirs, so `bool_type2t` and
  `empty_type2t` also receive spellings and **still drop them**. That is left alone deliberately:
  `cpp_expr2string` handles neither string, `goto2c` reads only the bitvector kinds, and
  `id2string(bool_typet().id())` already equals `"bool"` so the exception-id fallback coincides. It is
  not a claim that no other kind is ever spelled.

The field is **unreflected**, like `argument_base_names` (§44), `member_base_names` (§46), `cformat`
(§69) and `constant_qualified` before it: a spelling is no part of a type's identity, so two bitvectors
of a width stay the same type however they were spelled, and the unit test asserts that on `==` and
`crc()` both.

Why each arm is carried differs, and only one of the three answers is "a Python consumer reads it":

```
signedbv    is_char_type (type_utils.h:207) -- the only Python reader of the spelling
unsignedbv  the same reader under char_is_unsigned, plus goto2c/expr2c.cpp:174
floatbv     no Python reader at all: is_char_type tests the bitvector kinds only,
            get_python_type_category branches on is_floatbv() rather than the
            spelling, and python printing goes through c_expr2string, which never
            reads it. Carried for cpp_expr2string.cpp:166 and the exception-id
            path, and pinned by the unit round-trip rather than end-to-end.
```

That last row matters for what a future regression would catch: dropping **only** the floatbv carry
fails the unit test and nothing else.

A third reader the first draft missed, and it is not a printer:
`clang_cpp_exception_id.cpp:45` feeds throw/catch id matching, and
`clang_cpp_adjust_irep2.cpp:147`/`:161` call it on `migrate_type_back` output -- so it sits directly in
this change's blast radius. `esbmc-cpp/try_catch` is 172/172, which discharges it empirically.
`python_adjust.cpp:1051`'s comment claimed the attribute never survives migration and that Python types
never carry it; both halves were false after this change and are corrected there.

### 10.2 The evidence, which §9's attempt did not have

```
base -- no carry, no writes                     VERIFICATION SUCCESSFUL
the eleven writes, no carry                     VERIFICATION FAILED      (§9)
the eleven writes, with the carry               VERIFICATION SUCCESSFUL
with the carry mutated out of migrate_type      VERIFICATION FAILED
```

`regression/python/github_4715_cpp_type_char{,_fail}` pins it, and both halves are gates. The
SUCCESSFUL half fails with `assertion 0` when the forward carry is removed. The `_fail` half needed
help to be one: `^VERIFICATION FAILED$` alone holds either way, so it also pins the *claim shape* --

```
assertion (signed int)((signed char)val) == (signed int)((signed char)({ 120, 0 }[0]))
```

-- which a dropped spelling collapses to `assertion 0`. That is the difference between a test that
records the verdict and a test that records why.

The seven CORE tests §9 named now pass: `casting14`, `enumerate8`, `for-loop3`, `for-loop6`,
`for-loop8_fail`, `python_irep2_adjust_only_string_index`, `string-concat6`.

Because the change is in `irep2_type.h` it is global, so the breadth matters, and each figure names a
command a reader can run:

```sh
ctest -LE regression                      884/884   (883 before this change adds its own case)
ctest -L esbmc-solidity                   526/526
ctest -L esbmc-cpp/cpp                    1065, six failures already failing on master (§2726)
ctest -R "regression/esbmc-cpp/try_catch" 172/172   the exception-id reader above
ctest -R "irep2"                          238/238   190 of them in the core C suite
```

`fields_cover_class` accepts the new field on all three kinds -- but only just, and that is worth
recording rather than celebrating. Measured: `sizeof` goes 48 -> 56 on the two bitvector kinds, leaving
`derived - covered` at exactly the `alignof - 1` budget of 7. The `irep_idt` pushes `constant_qualified`
into a fresh eight-byte slot and leaves four bytes of genuine trailing padding, so a further unreflected
field would fit in the hole **without** tripping the guard -- verified by an A/B on two header trees:
adding a spare `unsigned int` fails the static assert before this change and passes after it. No
declaration order avoids it, since `width + cpp_type + bool` cannot fit in eight bytes. Each bv kind
therefore now pins its own layout with a `static_assert`, so the next field has to come through that
comment first. This is the second time the repo has been bitten around `fields_cover_class`.

The eight bytes cost nothing measurable in time, which is worth having checked rather than assumed for
the most-constructed nodes in the tool. `ESBMC_REGRESS_TIMEOUT_MAX=45 ctest -L esbmc-solidity` reports
one test over budget, `mul_cnt_ver_2`; standalone it takes **39.90 s with the carry against 39.99 s
without**, so it is the known `-j4` contention artefact rather than a regression, and it passes
uncapped.

Python B-2* 54 -> 43; repo total 125 -> 114.

### 10.3 What this does not settle

§8.1 asked for something else: give IREP2 the distinction, on the ground that a Python character is not
a spelling detail and `unsignedbv` of width 8 is the wrong model for it. This change does not do that.
It carries the spelling, which unblocks the writes and pins the behaviour, and leaves the type-model
question exactly where §8.1 put it. The carry is compatible with either answer -- if the distinction is
later given its own kind, the field becomes redundant and can go -- but it should not be read as having
decided the question.

Two of `python_expr_builder.cpp`'s six hand-restorations (`:40`, `:71`, `:90`, `:112`, `:176`, `:312`,
each commented "migrate_type does not round-trip `#cpp_type`") are now redundant for these kinds and
could be removed; that is a separate change with its own measurement, not a rider.

## 11. The funcdef cluster: two blockers, neither of them `#cpp_type` (2026-09-17)

§3's table listed `converter_funcdef.cpp`'s writes as failing `class_var_param_augassign{,_fail}` with
the cause "same shape, parameter and return types" -- i.e. `#cpp_type`. With §10's carry in, they still
fail, so that attribution was wrong. Measured, and the two halves fail differently.

### 11.1 Split by half, because the whole tells you nothing

`get_function_definition` (`:2055-2548`) holds eight of the file's twelve writes: seven
`added_symbol->set_type(type)` over the function's own `code_typet`, and one
`added_symbol->set_value(function_body)`.

```
both halves converted        SIGSEGV
the body write alone         SIGSEGV
the seven type writes alone  VERIFICATION FAILED -- "assertion count == 5"
base                         both tests pass
```

Converting all eight and seeing one failure would have suggested one cause. They are two.

### 11.2 The body write must not be converted, and §8.2 says so

The crash is in `std::construct_at<irep_idt>` under `process_goto_program`, i.e. a consumer copying
names out of the symbol after conversion. The reason is already in this document: §6 is titled *a body
cannot be migrated before its symbols exist*, and §8.2 is explicit --

> The program-entry body writes (§6) are correct as they are: `set_value(const exprt &)` defers
> migration until the symbol table is complete, which is exactly what a body needs. Converting them is
> not blocked work, it is work that must not be done.

`:2539` is one of those writes. The B-2 bar counts it as residue because it counts the argument's
spelling, and that is the bar being wrong rather than the code. It should be struck from the residue
rather than left looking like debt -- the same correction §8.2 already made for two writes and this one
escaped.

### 11.3 The type writes are a real blocker, and not this attribute

The seven type writes produce a wrong verdict, not a crash: `class_var_param_augassign` asserts
`count == 5` after an augmented assignment to a class variable through a `Class*` parameter, and with
the function's `code_typet` stored IREP2-side that assertion fails. So a `code_typet` round trip loses
something this path needs, and it is not `#cpp_type` -- that is carried now, and the failure is
unchanged.

### 11.3.1 Measured: it is the argument's *plain* `identifier`

None of the three candidates this section first listed. Probing all seven sites with
`full_eq(migrate_type_back(migrate_type(type)), type)` reports one difference at site 7, on `advance` --
the function the failing test exercises. **That "one" is an artefact of the probe's input, not a property
of the seam**: `advance(c: Counter)` has no default argument, so the probe could not see a third key that
is also dropped. §11.3.3 has it. The measured diff on `advance` is:

```
ORIG argument 0                               BACK argument 0
  * type: pointer -> symbol tag-Counter         * type: pointer -> symbol tag-Counter
  * identifier: py:main.py@F@advance@c          (dropped)
  * #location: {file, line, function, column}   (dropped)
  * #base_name: c                               * #base_name: c
  * #identifier: py:main.py@F@advance@c         * #identifier: py:main.py@F@advance@c
```

So `#identifier` and `#base_name` both survive -- §44's carry works -- and what does not is the
**plain `identifier`**, plus the argument's `#location`. The `#location` loss is inert: a sweep of `src/`
finds no reader of an argument's `location()` at all, only the writer at `converter_funcdef.cpp:1678`. That is the same plain-versus-comment trap as
the struct component base name: `code_typet::argumentt::get_identifier()` returns `cmt_identifier()`
(`std_types.h:332-335`), so the seam reads and writes `#identifier` at both ends
(`migrate.cpp:352`, `:3167`) and never touches the plain key at all.

It also rules out the obvious reader. Symex takes parameter names from
`function_type.argument_names` on the **IREP2** side (`symex_function.cpp:185`), which is populated from
`#identifier` and therefore intact -- so this is not the `symex_function.cpp:219` unnamed-parameter skip.

### 11.3.2 The reader, and the fix that follows from it

Not identity after all -- a reader, found by searching for the key rather than guessing.
`converter_funcall.cpp:1537`:

```cpp
copy_instance_attributes(
  params[i].identifier().as_string(),     // the plain key the seam drops
  arg_sym->identifier().as_string());
```

Its own comment describes the failing test: *"if `o.x = 5` is set inside `f(a)` via parameter `o`, then
`a.x` should reflect the instance attribute rather than the class attribute."* That is
`c.count += c.step` inside `advance(c)`. With the plain key gone the first argument is `""`,
`copy_instance_attributes` propagates nothing, and `count == 5` fails. The chain end to end:

```
seam drops the plain identifier -> copy_instance_attributes("") is a no-op
  -> the instance attribute never reaches the caller's argument
  -> c.count += c.step is lost -> "assertion count == 5" fails
```

And the fix is the reader, not a carry. A python parameter sets both keys from one string
(`register_function_argument`, `converter_funcdef.cpp:1676-1677`):

```cpp
arg.cmt_identifier(arg_id);   // #identifier -- carried by the seam
arg.identifier(arg_id);       // plain       -- dropped
```

"Its only writer" would be the wrong justification, and was the first draft's: there are at least six
writers of an argument identifier across the frontends. The claim that holds is stronger and checkable
-- **no writer anywhere sets the plain key to a string different from `#identifier`, and none sets the
plain key without `#identifier` also being set to the same string**. `clang_c_convert.cpp:910` sets only
the `#` one; `clang_cpp_convert.cpp:2880` sets the plain one and reaches `:910` for the other with the
same value; `migrate_type_back` and the polymorphic-builtin path set only the `#` one. So `#identifier`
is a superset: equal where both exist, non-empty where the plain key is empty, never less informative.
That makes the change safe for any frontend's argument reaching either reader, including the
`func_symbol->get_type()` case where the callee could be a C operational-model function -- there the old
code read `""` and the new code reads the real id, so those callers are fixed rather than risked. Two readers now take
it -- `converter_funcall.cpp:1537` and `function_call/expr.cpp:6565` -- and the seven `code_typet` writes
land with `class_var_param_augassign{,_fail}` passing.

The two readers failed differently, which the chain above describes for only one of them.
`copy_instance_attributes` keys on a `find` and returns, so an empty key is a no-op. But
`element_type_registry::assign_from(from, to)` writes `map_for(slot)[to]` unconditionally once `from`
resolves, and the lost key is the **destination** -- so pre-fix that reader wrote the caller's element
types under the empty key, polluting the registry rather than doing nothing. The fix repairs a bad write,
not just a missing one.

### 11.3.3 A third key, found by probing an input the first probe did not have

The seam drops **three** keys per argument, not two. Re-probing on a function that has a default --
`def f(a, b=5)` -- adds:

```
  * #default_value: constant 5      (dropped)
```

Structural, not an oversight in the carry: `code_type2t`'s fields are `arguments`, `ret_type`,
`argument_names`, `ellipsis` and the unreflected `argument_base_names` (`irep2_type.h:326-334`). There is
no slot for a default, and neither direction of `migrate_type` touches one.

Three readers consume it, and two take it off the **function symbol's** type:

```
function_call/expr.cpp:6767    finalize_call, off func_symbol->get_type()
converter_funcall.cpp:1025     off resolved_func_type = &to_code_type(target_func->get_type())
converter_funcall.cpp:1379     decides whether to raise TypeError: missing required positional argument
```

That matters because `python_adjust.cpp:86-89` already states the invariant:

> a resolved-alias code type written back here carries no argument `default_value` (the attribute does
> not survive the IREP2 round-trip) -- default arguments must be sourced from the function symbol, not
> from a variable's type.

On the seven converted paths the function symbol stops being that safe source.

**Latent, not live, and unpinned.** Thirteen probes failed to make the two conditions coincide: the
Python preprocessor normalises defaults into the AST before the converter sees a call
(`preprocessor/core_visitors_mixin.py:1413-1452`), so shapes that reach a reader fire no write and shapes
that fire a write never reach a reader. So no verdict moves today, and
`class_var_param_augassign{,_fail}` structurally cannot observe it. That is a reason to record it, not a
reason to call it safe -- a change to the preprocessor's normalisation would expose it with nothing in
the suite watching. Carrying `#default_value` is its own change with its own oracle: a default-argument
test, which the suite does not currently have in a shape that reaches these writes.

This is the first of the six seam losses where §80's fix-the-reader route applied, and it is worth
contrasting with §10 on cost. The carry there bought one attribute for eight bytes on the three
most-constructed type kinds and all of `fields_cover_class`'s remaining margin. This bought one for a
changed accessor at two call sites. Nothing about the plain key made `copy_instance_attributes` prefer
it; it was simply the one reached for first.

### 11.4 Standing

Python B-2* 43 -> 37; repo total 114 -> 108, measured with `python3 scripts/irep2/bars.py`, and
`ctest -R regression/python/class_var_param_augassign` 2/2.

The pair pins the **reader fix**, not the conversion: both halves fail without it. Nothing fails if the
six writes are reverted, so the bar move is the only evidence for that half -- inherent to a migration
whose contract is "behaviour unchanged", but worth saying rather than leaving a reviewer to infer. The six
sites are each reached by pre-existing CORE tests: `github_4373_nested_def` (2328), `return5` (2339),
`github_4514`/`github_4352`/`github_7085` (2361), `sv_verifier_nondet_list`/`github_4744_fail` (2411),
`optional7`/`tuple18`/`github_3846_3` (2495), and `class_var_param_augassign` (2518). The funcdef cluster is not ten
writes blocked on one attribute, as §3 had it. It is **six converted** and two that stay legacy: the body write at `:2539`
by §8.2's rule -- a body cannot be migrated before its symbols exist -- and the arm at `:2439`, which no
test in the suite reaches and whose guard re-runs the same
`infer_return_type_from_body` the arm ninety lines above already ran, so it is a C-Dead candidate rather
than a conversion. Converting an unexercised line is how `:2539`'s SIGSEGV was found.

On the body write, "B-2 should stop counting it" is an ask on `scripts/irep2/bars.py`, which still counts
it -- the 36/107 figures above include it. Recorded as a proposal, not as an applied exclusion; until the
script implements §8.2's category the next survey re-adds the write as debt.

That makes it the sixth marker found not to survive the seam, and the second where the plain key and the
`#` key were confused for one another. Worth stating as a check rather than a story: when a legacy node
carries both `x` and `#x`, establish which one each accessor reads before concluding a carry covers it.
`argumentt::get_identifier()` reads `#identifier` while the node also holds a plain `identifier`, so
§44's carry looked complete and was not.

## 12. `bases`: the write is the repair, so it stays (2026-09-17)

§3's table listed `python_adjust.cpp:70` as failing a unit case, with the cause "the write exists to
re-attach the legacy-only `bases` sub-irep, and storing IREP2 drops it again". That is right, and the
conclusion it implies is worth making explicit: this write must stay legacy, and the question it raises
is structural rather than a conversion.

### 12.1 What the site does

```cpp
const irept bases = symbol->get_type().find("bases");
symbol->set_type(t);                     // IREP2 -- drops `bases`
if (bases.is_not_nil())
{
  typet patched = symbol->get_type();    // back-migrated view
  patched.set("bases", bases);           // re-attach
  symbol->set_type(std::move(patched));  // <- the residue B-2 counts
}
```

The legacy write *is* the compensation for the loss two lines above it. Converting it would drop `bases`
a second time and undo the repair, which is why the repo's own unit case
(`python_adjust pre-pass write-back preserves bases for the throw chain`) catches it -- and why §3 notes
that a scripted conversion walking into it is the argument for the test rather than against the script.

### 12.2 All three routes, and why none is cheap

§80 gives three answers when a marker does not survive the seam. For `bases` each is priced:

- **Derivation** is unavailable. clang-cpp records inheritance structurally as `@base@<class_id>`
  components, so a base list is recoverable there -- but the Python frontend stores it *only* in the
  sub-irep (`python_class_builder.cpp:83`, `st.add("bases").get_sub()`), and emits no `@base@`
  component. There is no second source to derive from.
- **Fixing the readers** is not one frontend's work. `find("bases")` has four readers in four layers:
  `util/expr/base_type.cpp:421`, `clang-cpp-frontend/clang_cpp_exception_id.cpp:27`,
  `python-frontend/python_adjust.cpp:1088` and `goto-programs/remove_exceptions.cpp:718`. They would all
  need another source, and by the point above there is none.
- **Carrying it** cannot use the unreflected pattern the other five carries used. `bases` has no `#`, so
  it lives in `named_sub` and takes part in `typet` equality (§3.1) -- a faithful IREP2 field would have
  to be *reflected*, changing `struct_type2t`'s identity and therefore every hash and cache keyed on a
  struct type. That is a different order of change from §10's eight bytes.

### 12.3 So it is a decision, not a task

`python_adjust.cpp:70` joins `:2539` and `converter_stmt.cpp:1351` as a write that stays legacy for a
stated reason, and B-2 counts all three only because it counts the argument's spelling. What is left is a
structural question of the same kind as §8.1 and the Phase 6 trio: **should inheritance cross the seam as
a reflected field on `struct_type2t`, or should Python record bases structurally as clang-cpp does, so the
list becomes derivable?** The second is the smaller change to the IR and the larger one to the frontend,
and it would retire this residue and the exception-id divergence §11.3.1 notes together. Neither should
be picked without the maintainers, and neither is blocked on measurement -- the routes above are priced.
Python B-2* stays 43; repo total 114. The funcdef cluster is not ten writes blocked on one attribute,
as §3 had it. It is one write that must stay legacy and seven blocked on an unidentified `code_typet`
loss, and the next step is the three experiments above rather than another conversion attempt.
So Python's B-2 residue stays 54, and the next task is the carry itself, with a regression pair over
`val = "hello"[0]; assert val == "h"` added in the same change so a later attempt at these eleven cannot
pass review silently.
