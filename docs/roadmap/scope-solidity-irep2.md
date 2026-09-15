# Scope: solidity frontend to IREP2 (Phase 8)

Opened per `frontends-to-irep2.md` §6, which requires each of Phases 5-9 to
start with its own scope doc: census, phased decomposition, gates, risks. Phase
5 (jimple) closed at `scope-jimple-irep2.md` §31; Phase 6 (clang-c) is
`scope-clang-c-irep2.md`; Phase 7 (clang-cpp) is `scope-clang-cpp-irep2.md` and
is **not** closed, which matters here for the reason §2 gives.

## 1. Census

### 1.1 The blocker the parent records is stale

`frontends-to-irep2.md` §15.1 records Solidity as *blocked, not zero*: a decline
census got `ERROR: `' is not a goto-binary`, nothing ran, and the note concludes
the suite needs Linux CI. Re-tested 2026-09-11 and it is measurable here.

```
$ grep ENABLE_SOLIDITY build/CMakeCache.txt
ENABLE_SOLIDITY_FRONTEND:BOOL=On
$ ctest -j24 -L esbmc-solidity
99% tests passed, 2 tests failed out of 525
```

The two are not failures: both print `passed but is marked as KNOWNBUG.
Consider reclassifying it as CORE` (`testing_tool.py` exit 77) —
`delegate_shadow_3` and `nested_array_deep_1`, each expecting `^VERIFICATION
SUCCESSFUL$` and getting it, in half a second, so not the timeout that
`KNOWNBUG` silently accepts. They are master drift, and flipping them wants its
own change; they are recorded here because a census that reports "2 failures"
without saying which kind is the §15.1 mistake again.

§15.1's own rule is what makes this census admissible: it must show the thing
under test executed before a zero means anything.

### 1.2 Surface

`src/solidity-frontend`, 23 599 LOC over 25 files:

| File | LOC | Legacy construction sites |
|---|---:|---:|
| `solidity_convert_call.cpp` | 3 682 | 404 |
| `solidity_convert_expr.cpp` | 3 409 | 250 |
| `solidity_convert.h` | 1 129 | 221 |
| `solidity_convert_ref.cpp` | 937 | 89 |
| `solidity_convert_type.cpp` | 1 588 | 86 |
| `solidity_convert_constructor.cpp` | 1 055 | 78 |
| `solidity_convert_stmt.cpp` | 978 | 75 |
| `solidity_convert_decl.cpp` | 1 526 | 73 |
| `solidity_convert_mapping.cpp` | 708 | 70 |
| `solidity_convert_contract.cpp` | 850 | 67 |
| `solidity_convert_tuple.cpp` | 653 | 59 |
| `solidity_convert_builtin.cpp` | 524 | 54 |
| `solidity_convert_modifier.cpp` | 752 | 53 |
| `solidity_convert.cpp` | 1 266 | 51 |
| `solidity_convert_util.cpp` | 1 046 | 36 |
| `solidity_convert_literals.cpp` | 159 | 15 |
| `solidity_grammar.cpp` | 1 804 | **0** |
| `pattern_check.cpp` | 153 | **0** |
| remainder (`typecast`, `language`, headers) | — | 4 |
| **total** | **23 599** | **1 685** |

"Construction sites" counts lines mentioning `exprt`, `typet`, a `code_*t`,
`symbol_expr`, `gen_zero` or `from_integer` — the same rough proxy the parent
used for its 971 and 1 420 figures, not a precise obligation count. The
parent's 1 420 was measured at `f14cd73ff8`; the surface has grown since.

**IREP2 use today: zero.** `grep -c 'expr2tc\|type2tc' *.cpp *.h` over the
whole frontend returns 0, so unlike clang-c (49 sites already native) this
phase has no head start. `solidity_grammar.cpp` is the AST-kind classifier and
constructs no IR at all — the analogue of `clang_c_lexer.cpp` in Phase 6's
§1.1, and it needs no migration.

### 1.3 Corpus

| Suite | Tests |
|---|---|
| `esbmc-solidity` | 525 |

Tests ship a pre-generated `contract.solast` beside `contract.sol`, and the
flags line names both (`--sol contract.sol --contract <Name>`), so `solc` is
not needed to run them. That is why the suite is measurable without the
toolchain §15.1 assumed was required.

## 2. The finding that sets this phase's shape: there is no Solidity adjust pass

`solidity_languaget::typecheck` (`src/solidity-frontend/solidity_language.cpp`)
has four phases — intrinsics, the `sol64` operational models, the converter,
then the adjuster — and the adjuster it runs is **`clang_cpp_adjust`**, Phase
7's subject, not one of its own:

```cpp
  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;
```

Two consequences, and they pull in opposite directions.

**The adjust half of this phase is inherited, not owed.** Every arm Phase 7
ports serves Solidity at no extra cost, and none of Solidity's 1 685 sites is
in an adjust pass. Phase 8's own work is entirely in the *converter*.

**Solidity is therefore exposed to every Phase 7 divergence, and is not
currently measuring any of them.** `clang_cpp_language.cpp` gates its adjuster
on `clang-cpp-irep2-adjust-only`; the line above does not, so the flag does not
reach this path — a Solidity test run with it is byte-identical to one run
without (checked on `abi_decode_1`; the claim rests on the source, since one
identical verdict would not distinguish "flag ignored" from "flag made no
difference here").

That makes the first action of this phase cheap and valuable: wire the existing
flag through, and the 525-test Solidity corpus becomes a **second, independent
corpus for Phase 7's pass** — one whose input the C++ frontend never produces,
over a converter with different habits. Phase 7's divergence census has been run
on `esbmc-cpp` only.

### 2.1 The save/restore dance is a seam hazard

Phase 4 of that function saves every library symbol's value as an `exprt`
before adjusting and restores it afterwards, because `clang_cpp_adjust` would
corrupt bodies that `c2goto`'s `clang_c_adjust` already adjusted.
`clang_c_adjust_irep2` writes back only values it changed, and writes them back
through `migrate_expr_back`. So the restore interacts with a different
write-back discipline than it was written against, and any seam loss on a
library body would be masked by the restore rather than observed. This wants a
measurement before the flag is wired, not after.

## 3. Proposed decomposition (not yet executed)

1. **S.1** Wire `clang-cpp-irep2-adjust-only` into
   `solidity_languaget::typecheck` and report the divergence count over the 525
   tests. Measured baseline, no porting. Gated on §2.1's measurement.
2. **S.2** A read-only `migrate` census for the Solidity converter's output, as
   `--clang-cpp-irep2-migrate-census` is for C++: run every value through
   `get_value2()` and report what `migrate_expr` cannot represent. This is the
   only way to price the converter before touching it, and it is the step
   §15.1's void figure was trying to be.
3. **S.3** The converter's construction sites, in the order the census ranks
   them. `solidity_convert_call.cpp` and `solidity_convert_expr.cpp` are 39 %
   of the surface between them.
4. **S.4** Remove the legacy path once S.1's divergence count is zero.

Steps S.1 and S.2 are both measurement, and both are cheap. Neither is
committed to a porting order, deliberately: Phase 6 §60 found the adjuster's
arms to be one strongly-coupled component that could not move singly, and Phase
7 §3 found its pass was not extensible in the way Phase 6 assumed. A census
first is the lesson those two paid for.

## 4. Gates

The parent's §7 gates apply unchanged. Two are worth restating for this phase:

- **A census must show the thing under test executed.** §15.1's "14 tests, 0
  declines" is void because nothing ran. Any figure this doc reports names the
  command that produced it.
- **`KNOWNBUG` and `FUTURE` rows cannot be read as green.** They accept a
  timeout as satisfying the expectation, so a Solidity divergence count must
  separate them out; `grep` for `accepted under KNOWNBUG` before reading a run
  as clean.

## 5. Risks

| # | Risk |
|---|---|
| R1 | This phase's adjust half depends on Phase 7 closing. Wiring the flag before Phase 7's divergences are down exposes Solidity to all of them at once, which is why S.1 is a measurement and not a flip. |
| R2 | The §2.1 restore can mask a seam loss on a library body. A divergence count taken without checking it may be optimistic. |
| R3 | The `sol64` operational models are a second `c2goto`-compiled artefact, adjusted by `clang_c_adjust` at build time. The `building-c-library` exemptions that protect the C models have no Solidity analogue recorded, and Phase 6's name-matched-builtin section found one such exemption already unreachable from the IREP2 pass. |
| R4 | 1 685 sites and zero head start make this the second-largest phase after Python. Its ordering before Python is the parent's §215 judgement and this doc does not revisit it. |
| R5 | The two XPASS rows in §1.1 mean the suite's expectations are drifting from master. A divergence count is only meaningful against a suite whose baseline is green. |

## 6. Next

S.1 and S.2, in that order, each as its own change. Neither ports anything.

## 7. B-2 censused by measurement, and the wall it hits (2026-09-15)

`git grep 'set_type(\|set_value(' -- src/solidity-frontend | grep -vc 2tc` is **100**:
**23** type writes and **77** value writes, spread over 13 files, with
`solidity_convert_call.cpp` (31) and `solidity_convert_decl.cpp` (16) holding half.

### 7.1 Converting all 23 type writes fails 151 of 525

Measured, not predicted. The first failure names its own cause:

```
esbmc: solidity_convert_expr.cpp:1689:
  get_contract_member_call_expr(...): Assertion `!base_cname.empty()' failed.
```

`base_cname` comes from `get_sol_contract(base.type())`, i.e. the `#sol_contract`
attribute on a *variable's type*. `migrate_type` drops it, so every contract-member
call through a variable loses the contract it belongs to.

That is not one attribute. The frontend keeps eleven of them on legacy `typet`
nodes, each with a getter beside the setter in `solidity_convert.h`:

```
#sol_array_size  #sol_bytesn_size  #sol_contract   #sol_data_loc
#sol_dynarray_state  #sol_mapping_array  #sol_name  #sol_state_var
#sol_type  #is_sol_virtual  #is_sol_override
```

All eleven are read only inside `src/solidity-frontend`, and none has an IREP2
field. So Phase 8's B-2 is not a site-by-site job: a Solidity symbol's *type* is
where the frontend keeps its Solidity-level meaning, and IREP2's closed type system
models none of it. This is the same shape as §46's `member_base_names` and the C++
exception specification, at eleven times the size.

### 7.2 What did convert: the seven function types

Seven of the 23 write a *function* type for a synthesised function -- the contract's
`main`, a modifier's wrapper, a constructor, an auxiliary call helper -- and those
carry no `#sol_*` attribute. With them on `migrate_type`, `esbmc-solidity` is
**525 of 525** and the unit suite 876 of 876.

That is the whole tractable subset of the 23 until the eleven attributes are decided.
The remaining sixteen write a variable's, a mapping's or a contract's type, which is
exactly where the attributes live.

### 7.3 What this asks of the plan

The decision is the same one Phase 7 now waits on, and Solidity makes it sharper: an
unreflected field per attribute does not scale to eleven, and a generic
leftover-`irept` carrier is the escape hatch B-4 forbids (§47.3). The third option
the C++ side has used twice -- derive the value from something IREP2 already holds
(§47, §50) -- needs checking per attribute: `#sol_contract` on a variable's type is
recoverable from the contract symbol the variable belongs to, `#sol_bytesn_size` from
an array type's size, and so on. Those are eleven small questions rather than one big
one, and none of them has been asked yet.

## 8. `#sol_contract` retired by derivation: one of the eleven, and the method (2026-09-15)

§7.3 asked whether each of the eleven type attributes can be derived from something
IREP2 already holds. Here is the first answer, and it is yes.

### 8.1 The derivation

`#sol_contract` is written in exactly one place, and the line above it says what the
type is:

```cpp
new_type = pointer_typet(symbol_typet(prefix + cname));
set_sol_type(new_type, SolidityGrammar::SolType::CONTRACT);
set_sol_contract(new_type, cname);          // removed
```

The contract name *is* the symbol-type identifier with `prefix` ("tag-") removed, and
IREP2 keeps both the pointer and the symbol type's identifier. `prefix` is also used
for structs, so the shape alone is not sufficient; the converter already holds every
contract's name in `linearizedBaseList`, which supplies the rest. `get_sol_contract`
now computes it and `has_sol_contract` asks it, the setter and the attribute are
gone, and `esbmc-solidity` is **525 of 525**.

### 8.2 What it unblocked, measured

Converting all 23 type writes with the attribute still in place fails **151** of 525
(§7.1). With it derived, the same 23 fail **132**. So this one attribute accounted
for 19 tests, and the next failure is a different cause -- a `CONVERSION ERROR`
naming a `tag-Base` subtype, i.e. one of the remaining ten.

That is the shape of the rest of the phase: derive an attribute, re-run the
all-23 experiment, see how far it gets, and keep the conversions that stay green. The
number to watch is the failure count, not whether any single site converts.

### 8.3 The membership test is unpinned, and probably unfalsifiable from source

Dropping `linearizedBaseList.count(cname)` -- so that any pointer to a `tag-` symbol
type reports a contract name -- leaves the suite at 525 of 525. The guard's own
consumer is `get_contract_member_call_expr`, reached only for a *member call* on the
base, and Solidity structs have no member functions, so a struct-typed base appears
not to reach it at all. The test is kept because it is the precise condition rather
than because a test pins it, and no test was invented to cover a case the language
may not admit. `struct_3`'s `book.book_id` is a field access and does not go through
that path.

## 9. The eleven censused by reader count, and three of them are dead (2026-09-15)

§8 derived one. Counting writes and reads per attribute -- across `src/` and `unit/`,
not only the solidity frontend -- splits the rest into three groups rather than ten
equal problems:

| attribute | writes | reads |
|---|---|---|
| `#sol_type` | 60 | 51 |
| `#sol_array_size` | 10 | 17 |
| `#sol_bytesn_size` | 9 | 15 |
| `#sol_mapping_array` | 4 | 9 |
| `#sol_dynarray_state` | 3 | 8 |
| `#sol_name` | 4 | 3 |
| `#sol_state_var` | 4 | 3 |
| `#sol_data_loc` | 5 | **0** |
| `#is_sol_virtual` | 2 | **0** |
| `#is_sol_override` | 2 | **0** |

### 9.1 Three go away outright

`#sol_data_loc`, `#is_sol_virtual` and `#is_sol_override` have **no reader anywhere
in the tree**. `solidity_convert.h` even documented the first as "Set-only today (no
readers)". Removed, with their setter and the two `if`/`else if` chains whose only
bodies they were: `esbmc-solidity` 525 of 525, unit 876 of 876.

The instrument for a removal like this is not a reachability proof -- the branches
were reachable, their *effect* was unobservable -- so the argument is the grep: zero
readers tree-wide means nothing can distinguish the write from its absence. Nothing is
lost that cannot be recovered either: the data location is read straight off the
AST's `storageLocation` at three other places in the frontend, and `virtual` /
`overrides` off the AST node the write sat next to.

### 9.2 `#sol_type` is not derivable, and that decides the phase's shape

60 writes and 51 reads, holding a `SolidityGrammar::SolType` -- `ADDRESS` against
`UINT160`, `BYTES` against an array, `CONTRACT` against a struct. Those are exactly
the distinctions IREP2's type system normalises away, so there is nothing to derive
it from. `#sol_contract` was derivable because the contract name was still spelled in
the symbol type's identifier; a SolType is not spelled anywhere else.

So the remaining seven live attributes do not all have §8's answer, and the phase
needs one more option than "derive it" or "add a field". The one that fits the bars:
a **frontend-owned side table** keyed by symbol id, holding what is Solidity-level
rather than representation-level. It keeps IREP2 closed (B-4: no attribute escape
hatch on the shared representation), keeps the information where its only readers
are, and needs no seam carriage at all -- a symbol's id survives every migration by
construction.

That is a proposal, not a measurement, and it should be argued before it is built.
The five attributes with few writes (`#sol_name`, `#sol_state_var`,
`#sol_dynarray_state`, `#sol_mapping_array`, and the two size attributes) are worth
trying §8's derivation on first, since each one that goes that way is one fewer entry
the side table has to hold.

## 10. §9.2's side table, built for one attribute (2026-09-15)

§9.2 proposed a frontend-owned side table for the attributes that are not derivable,
and said it should be argued before being built. Here it is built for the smallest
live one, so the argument has something measured under it.

`#sol_state_var` had one reader -- `solidity_convert_constructor.cpp:476`, asking per
declaration whether a variable is a contract state variable -- and two writers. It is
now a `std::unordered_set<irep_idt>` on the converter, keyed by the variable's symbol
id:

```cpp
std::unordered_set<irep_idt> sol_state_vars;
void set_sol_state_var(const irep_idt &symbol_id, bool v);
bool get_sol_state_var(const irep_idt &symbol_id) const;
```

The reader already had the id in hand -- it calls `context.find_symbol(comp.identifier())`
two lines further on -- and the writer needed moving 35 lines down `get_var_decl`, to
after `get_var_decl_name` computes it. `esbmc-solidity` is **525 of 525**, unit 876 of
876.

### 10.1 What it costs and what it settles

The key is a symbol id, which survives every migration by construction, so nothing has
to be carried across the seam and IREP2 gains no field. That is the whole point: the
information is Solidity-level, its only reader is in this frontend, and B-4 forbids
parking it on the shared representation.

The cost is that a reader must hold a symbol, not just a `typet`. For
`#sol_state_var` that was already true. It will not be true for every attribute --
`#sol_array_size` is read off `rt.subtype()` in places (`solidity_convert_util.cpp:710`),
where there is no symbol to key on -- so the table is not a universal answer, and
those readers would need the derivation of §8 instead, or a different key.

### 10.2 The running measurement

Converting all 23 of the phase's non-IREP2 type writes, at each stage:

| state | failures of 525 |
|---|---|
| attributes as they were (§7.1) | 151 |
| `#sol_contract` derived (§8) | 132 |
| `#sol_state_var` in the side table | **117** |

Each attribute retired is worth fifteen to twenty tests, and the three dead ones (§9.1)
cost nothing to remove. The number to drive to zero is that failure count; when it is
zero, all 23 type writes convert and Phase 8's B-2 is half done by count and most of
the way by difficulty, since the 77 value writes then have only §52's namespace
precondition between them and IREP2.

## 11. A third answer: read it off the AST (2026-09-15)

`#sol_name` fits neither §8's derivation nor §10's side table, and saying why is the
useful part.

It carries which Solidity spelling produced a call. `require`, `revert`,
`__ESBMC_assume` and `__VERIFIER_assume` all lower to the symbol
`c:@F@__ESBMC_assume` (`solidity_convert_ref.cpp:292-295`), so the symbol id does not
distinguish them -- which rules out deriving the value from the expression *and* keying
a side table by symbol, since the distinction is per call site rather than per symbol.

The reader is inside `get_call_expr`, which still holds `callee_expr_json` -- the same
AST node the writer read `blt_name` from one call deeper. So the name is read from the
AST at the point of use, and the attribute, its setter and its getter are gone.

`esbmc-solidity` is 525 of 525. Forcing the read to come back empty fails `error_1`
and `error_3`, the two `revert` tests, so the read is covered rather than merely
compiled. Only two of 525 move, which is itself worth knowing: 59 tests use
`require`, and its arm only drops a second argument that almost none of them passes.

The read sits in a one-line helper rather than inline, because `get_call_expr` is at
CCN 79 and the complexity gate blocks any increase over the threshold -- a ternary in
the function body took it to 80 and failed the gate.

### 11.1 The three answers, and how to choose

| when | answer | example |
|---|---|---|
| the value is still spelled in the IREP2 type | derive it | `#sol_contract` (§8) |
| it is a property of a symbol | side table keyed by symbol id | `#sol_state_var` (§10) |
| it is a property of a *syntactic site* | read the AST node at the point of use | `#sol_name` (§11) |
| nothing reads it | delete it | §9.1's three |

The third answer is the cheapest of the three when it applies, because the AST is
already in scope wherever the frontend is still converting -- and it applies exactly
when the attribute was a way of carrying AST information forward to a later point in
the same conversion. That is worth checking first for each remaining attribute, before
reaching for a table.

## 12. The seven are not independent: `#sol_type` is the root (2026-09-15)

§11.1 gave four routes and implied the remaining attributes could be taken in any
order. Taking the smallest next -- `#sol_dynarray_state`, three writes -- shows they
cannot.

### 12.1 What `#sol_dynarray_state` actually is

One writer, and it says so in one line:

```cpp
bool is_dynarray_state = get_sol_type(t) == SolType::DYNARRAY &&
                         is_state_var_check && !is_new_expr &&
                         !get_sol_mapping_array(t);
```

So the flag is a conjunction of three other facts. Two of them are now cheap: the
state-variable half is §10's side table, and `#sol_type == DYNARRAY` is the attribute
itself. The third, `!is_new_expr`, is a property of the declaration site and is gone
by the time any reader asks.

Its four readers pair it with the same companions -- `solt == DYNARRAY &&
base.is_symbol() && get_sol_dynarray_state(base.type())` at
`solidity_convert_ref.cpp:484` -- so the flag is largely re-deriving what its context
already establishes. But "largely" is not "exactly", and the missing piece is
`#sol_type`.

### 12.2 Every remaining attribute sits next to a `set_sol_type`

Walking each writer and looking three lines either side:

| attribute | writers | writers with a `set_sol_type` beside them |
|---|---|---|
| `#sol_array_size` | 8 | 6 (ARRAY, ARRAY_LITERAL) |
| `#sol_bytesn_size` | 5 | 2 (BYTES_STATIC) |
| `#sol_mapping_array` | 2 | 2 (DYNARRAY) |
| `#sol_dynarray_state` | 1 | 1 (DYNARRAY) |

They are refinements of a SolType, not independent facts: a size *of an array*, a size
*of a bytesN*, a flag *on a dynarray*. Which means whatever answer `#sol_type` gets
decides the shape of the answer for the other four, and doing them first would be
building on a foundation not yet chosen.

### 12.3 So the order is forced, and `#sol_type` needs a fifth answer

`#sol_type` resists all four routes of §11.1: it is not spelled in the IREP2 type
(§9.2), it is read off subtypes and expression types where there is no symbol to key a
table by (§10.1), it is read long after the AST node is gone, and 51 readers mean it is
not dead.

The option not yet tried is that it is *mostly* shape-derivable and only partly not.
`ARRAY` is an array type, `DYNARRAY` an infinite array, `CONTRACT` a pointer to a
contract tag (§8 already relies on that), `BOOL` a bool, and every `UINT<n>` / `INT<n>`
is a bitvector of width n. The kinds that genuinely collide are few -- `ADDRESS`
against `UINT160`, `BYTES_STATIC` against an array of bytes, `ARRAY` against
`ARRAY_LITERAL`.

That is a measurable claim rather than a hope: write a shape-based
`sol_type_from_type(const typet &)`, have `get_sol_type` compute both and log
disagreements, and run the 525-test suite. The disagreement set is then the real
residue, and only it needs a table or a field. That measurement is the next task, and
it should be done before any more of the four refinements are touched.
