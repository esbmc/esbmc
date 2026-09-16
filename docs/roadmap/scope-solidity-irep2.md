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

## 13. §12.3's experiment, run: `#sol_type` is not shape-derivable (2026-09-15)

§12.3 proposed writing a shape-based reconstruction of `#sol_type` and measuring the
disagreements. Instrumenting `get_sol_type` to record what it actually returns is
cheaper and answers the same question, so that is what was run: **26 170 observations
over 180 of the 525 tests**, 25 distinct kinds.

| kind | reads | | kind | reads |
|---|---|---|---|---|
| `UINT256` | 10 335 | | `DYNARRAY` | 487 |
| `<unset>` | 3 918 | | `BytesDynamic` | 389 |
| `ADDRESS` | 2 660 | | `BOOL` | 299 |
| `INT_CONST` | 2 468 | | `ARRAY` | 281 |
| `STRING` | 1 063 | | `ENUM` | 181 |
| `CONTRACT` | 927 | | `ARRAY_LITERAL` | 132 |
| `BytesStatic` | 881 | | `STRUCT` | 39 |
| `UINT8` | 694 | | `ADDRESS_PAYABLE` | 20 |
| `INT256` | 640 | | `ARRAY_CALLOC` | 18 |
| `MAPPING` | 511 | | `LIBRARY` | 1 |

And the IREP2 shapes those reads see: 16 628 `unsignedbv`, 3 285 `signedbv`, 2 840
`pointer`, 1 482 `symbol`, 594 infinite `array`, 315 `bool`, 208 sized `array`.

### 13.1 The collisions are the bulk of the traffic, not a residue

§12.3 guessed the ambiguous cases would be few. They are not:

| colliding class | reads | why the shape cannot tell them apart |
|---|---|---|
| `ADDRESS` / `ADDRESS_PAYABLE` vs `UINT160` | 2 680 | all `unsignedbv` of width 160 |
| `STRING` / `BytesStatic` / `BytesDynamic` | 2 333 | all arrays or pointers of bytes |
| `MAPPING` vs `DYNARRAY` | 998 | both infinite arrays |
| `ARRAY` / `ARRAY_LITERAL` / `ARRAY_CALLOC` | 431 | all sized arrays |

`BYTES1`..`BYTES32` are unsigned bitvectors of 8..256 bits, so each collides with a
`UINT<n>` as well. Over six thousand reads are in a class the type's shape cannot
separate. The hypothesis is refuted.

### 13.2 What that means for the phase, and the recommendation

`#sol_type` cannot be derived (§13.1), cannot be keyed by symbol -- 16 628 of the reads
are on an `unsignedbv` and 2 840 on a `pointer`, neither of which is a symbol's type
(§10.1) -- and cannot be read off the AST, since most reads happen long after the node
is gone. Deleting it is out: 25 kinds are live.

So B-2 for Solidity is bounded, and not by effort. A symbol's *value* is no better off
than its type: `migrate_expr` carries an expression's type through `migrate_type`, which
drops the attribute, so the 77 value writes hit the same wall as the sixteen type
writes. What converted (§7.2, the seven synthesised function types) converted because
those types carry no Solidity meaning at all.

The recommendation is to stop treating this as a migration task and record it as a
frontend design item: **Solidity's type system is not IREP2's, and today the gap is
bridged by attributes on the shared representation.** Three ways out, in increasing
cost and decreasing ugliness -- keep the attributes and accept that Solidity symbol
types stay legacy; give IREP2 a Solidity type kind, which the closed-type-system rule
exists to prevent; or carry a `SolType` beside every expression in the frontend's own
structures, which is the honest fix and a large refactor. Choosing among them is not a
measurement, and Phase 8 should not spend more ticks pretending otherwise.

## 14. Ten of Phase 8's writes, and the reader nobody had found (2026-09-15)

§13 ended by saying Phase 8 should stop pretending the `#sol_type` question is a
measurement. It is not, and the 91 remaining B-2 writes were left as though they all
waited on it. Ten do not. `scope-clang-c-irep2.md` §148.4's question -- what is the write
*for* -- converts them without touching a type, and answering it properly took two wrong
answers first.

### 14.1 The shape

Every one looks like this (`solidity_convert_call.cpp`, the external-call and transfer
wrappers):

```cpp
symbolt &added_old_sender = *move_symbol_to_context(old_sender);
code_declt old_sender_decl(symbol_expr(added_old_sender));
added_old_sender.set_value(msg_sender);            // <- counted as B-2 debt
old_sender_decl.operands().push_back(msg_sender);  // <- what goto_convert lowers
```

The value is written to the symbol and then written again as the declaration's operand.
Three are `msg_sender` / `msg_value` save-and-restore pairs around an external call, three
more are one half of that pair (`get_call_definition` and `get_staticcall_definition` save
only the sender, `model_transaction` only the value), and the tenth is the `$dl_success$`
flag's initial `false`.

### 14.2 The write is not redundant, and the suite cannot see why

It reads as a duplicate of the declaration's operand, and for *lowering* it is:
`goto_convertt::convert_decl` (`goto_convert.cpp:846-866`) takes the initialiser from the
decl's `op1` and never falls back to the symbol. Deleting the ten leaves
`regression/esbmc-solidity` at 525 of 525, which is what a first pass concluded from.

That conclusion was wrong. `mark_decl_as_non_det` (`mark_decl_as_non_det.cpp:31`) walks
every `DECL` and uses the symbol's value as its oracle for "was this declaration
initialised":

```cpp
    // Is the value initialized?
    if (s->get_value().is_nil())
      // Initialize it with nondet then
```

It is registered unconditionally (`esbmc/parseoptions/driver.cpp:407-409`), so with the
write gone each of the ten decls gains an `ASSIGN sym = NONDET(...)` between its `DECL` and
its real initialiser:

```
DECL unsigned _ExtInt(160) old_sender;
ASSIGN old_sender=NONDET(unsigned _ExtInt(160));   <- only without the write
ASSIGN old_sender=msg_sender;
```

The nondet is overwritten on the next instruction, which is exactly why 525 of 525 still
pass. The suite is blind to it; `--goto-functions-only` is not.

So the ten are **converted**, not deleted: `symbol_expr2tc` for the nine symbol
expressions and `gen_false_expr()` for the flag. The value stays non-nil, the pass skips
the decl as before, and B-2 is discharged with no change to the program. The flag's case is
exact by construction rather than by measurement: `migrate_expr_back` of
`constant_bool2t(false)` returns literally `false_exprt()` (`migrate.cpp:4436-4441`).

The nine are exact in the only channel that is read, not in every byte. `symbol_expr`
(`util/expr/expr_util.cpp:239-245`) sets the identifier *and* a cosmetic display name;
`symbol2t` carries only the identifier, and the type makes a `migrate_type` round trip. That
difference reaches no reader: the decl operand is untouched so the GOTO is unaffected, and
the one rendered channel for a symbol's value -- `show_symbol_table_plain` through
`c_expr2stringt::convert_symbol` -- resolves the symbol in the namespace by identifier and
prints the symbol table's own `name`, never the expression's.

### 14.3 What that costs, measured properly

Over the 515 of `regression/esbmc-solidity`'s 525 directories whose `test.desc` names a
source file that exists, the normalised `--goto-functions-only` dump is **identical for all
515**, and the hashes are 501 distinct values, so the comparison has content:

```sh
for d in regression/esbmc-solidity/*/; do
  src=$(sed -n 2p "$d/test.desc"); flags=$(sed -n 3p "$d/test.desc")
  [ -f "$d/$src" ] || continue
  h=$( (cd "$d" && esbmc "$src" $flags --goto-functions-only 2>&1) |
    sed -e 's/\x1b\[[0-9;]*m//g' -e 's/0x[0-9a-f]\{4,\}/0xX/g' \
        -e 's|esbmc_solidity_temp-[0-9a-f-]*|TMPDIR|g' |
    grep -vE '^WARNING: |^ *[0-9.]+s$|time: ' | LC_ALL=C sort | md5sum | cut -d' ' -f1 )
  echo "$h $d"
done | sort -k2
```

`scripts/irep2/test_bars.py` covers the script's refinement, including the two IREP2-only
builders this section needed it to recognise, and it now runs under ctest
(`unit/CMakeLists.txt`) rather than only by hand -- reverting any one refinement fails a
named case.

Two traps are worth naming because both produced a false "identical" first:
`--goto-functions-only` and `--symbol-table-only` write to **stderr** (stdout carries one
line, the version banner), so a capture piped `2>/dev/null` hashes nothing -- the tell is
that every program hashes the same, 1 distinct value across 515. And the Solidity frontend
extracts to a randomly named `/tmp/esbmc_solidity_temp-*` which appears in source
locations, so two runs of one binary disagree until it is normalised away.

### 14.4 The contract this exposes, which is worth more than the ten

A hand-built local declaration has to satisfy three readers, in three files, and no comment
says so:

| reader | applies to | reads |
|---|---|---|
| `goto_convert.cpp:846-866` (`convert_decl`) | non-static locals | the decl's operand |
| `clang_c_main.cpp:12-55` (`init_variable` via `static_lifetime_init`, gated on `s.static_lifetime`) | statics | the symbol's `value` |
| `mark_decl_as_non_det.cpp:31` | non-static locals | whether the symbol's `value` is nil |

For a static, `convert_decl` returns early (`goto_convert.cpp:833-836`) and the `push_back`
is the dead half; for a local, the `set_value` is not read for its content but is read for
its nil-ness. That is why `clang_c_convert.cpp:655-660` writes both for every initialised C
local, and why the pattern is a cross-frontend convention rather than Solidity debt.

It also retracts the discriminator an earlier draft of this section proposed. Deleting a
candidate group and running the suite does **not** sort live writes from dead ones here,
because the consequence of deleting a live one is a dead store the suite cannot observe.
The usable discriminator is the table above: a write whose symbol is `static_lifetime` is
read for its content, and one whose symbol is not is read for its nil-ness. Both are read.
On that reading none of this family is dead, and the family is convertible -- which is a
better outcome for the remaining 81 than a deletion sweep would have been.

With one caveat that does not apply to these ten and will apply to others. For a
`static_lifetime` symbol `init_variable` reads the value's *content* and emits it as a
`code_assignt` into `__ESBMC_main`, so there the display name dropped at the seam, and the
type's round trip, land in a rendered artefact rather than nowhere. All ten here are
non-static -- `old_sender` prints an empty `Flags` line and `$dl_success$` prints
`lvalue file_local`, and nothing in `solidity_convert_call.cpp` sets `static_lifetime` --
so the question does not arise yet. It arises at the first static one, and the answer has
to be measured on the GOTO rather than assumed from these ten.

### 14.5 What pins it

Nothing new can bite, and this time for a checkable reason: the conversion is
GOTO-identical over all 515 programs, and its one non-identity -- the display name dropped
at the migrate seam -- is rendered by no output channel and read by no pass. The place it
survives is the goto-binary irep (`symbolt::to_irep`), which no `test.desc` can reach,
since the harness matches output regexes only. What pins
it is that corpus comparison, `regression/esbmc-solidity` at 525 of 525, and the 49
Solidity contracts that use `msg.sender` -- 29 of them expecting `VERIFICATION FAILED`,
including `ext_call_state_track_2`, `reentrance_14` and `swc_107_2`, so the message context
this touches is pinned in both directions. Ten of those contracts show the converted value
directly: `--symbol-table-only` prints `Value.......: msg_sender` for each `old_sender`, and
that field can only be non-empty because the write ran.

Had the ten been deleted instead, a `--goto-functions-only` test pinning
`ASSIGN old_sender=NONDET` would have been owed and would have bitten. That test is the
reason to prefer the conversion, not a reason to add it.

## 15. The rest of the duplicated-initialiser family (2026-09-15)

§14 converted ten of these and left one question open: for a `static_lifetime` symbol
`init_variable` emits the value's *content* into `__ESBMC_main`, so what the migrate seam
drops is rendered there rather than ignored. §14.4 said that had to be measured rather than
assumed from ten non-static sites. This section measures it, converts sixteen more, and
finds two things §14 did not know about.

### 15.1 The first attempt at the answer was vacuous

`solidity_convert_decl.cpp:565` looked like the site to test -- it is the dynarray-state
arm, where `static_lifetime` is set when `is_dynarray_state` holds (`:346-349`). Converting
it gave an identical GOTO across all 515 programs, which answered nothing: instrumenting
the site shows **6 of 515 programs reach it and none with a static symbol**. The clean
comparison was a comparison of the non-static path.

`static_lifetime` is decided per *declaration* from the AST, not per site (`:346-349`: file
level, mapping, mapping-array, dynarray state, library constant), so no site is statically
one or the other and the only way to find one exercising both is to count. Every write in
the file, by flag:

```
site   runs static      site   runs static
:369    760     96      :565      6      0
:398      3      3      :665      4      4
:423     26     26      :695      4      0
:500      9      0      :710     15      0
:513     22      0      :718    845     12
:539      1      1      :1344    11      0
:545      2      2
```

### 15.2 The answer: the GOTO round trip is exact, the rendering is not

`:718` is the site that exercises both -- 845 runs, 12 with a static symbol. Converting it
leaves the `--goto-functions-only` dump identical for all 515 programs, so `init_variable`
emits the same assignment. §14.4's caveat is discharged for the GOTO.

It is not discharged for the symbol table, and that is the second thing §14 did not know.
Comparing `--symbol-table-only` instead -- which renders each symbol's value -- **12 of 515
programs differ, and all twelve differ only at `:718`**:

```
- this->x = 180374059643543449999388718682590567161426737540;
+ this->x = 0x1F9840A85D5AF5BF1D1762F925BDADDC4201F984;
```

Same number. `migrate_expr_back` rebuilds a constant through `integer2binary`
(`migrate.cpp:4408-4416`), so the literal's original spelling is discarded and the printer's
own default takes over; `a_hex_or_oct` is declared in `irep.h:1298` and appears nowhere in
`migrate.cpp`. So the round trip preserves the value and loses how it was written.

`:718` is therefore **not** converted here. The other sixteen are, and for those both
artefacts are identical across all 515 programs. Carrying the spelling across the seam --
the `argument_base_names` pattern of `frontends-to-irep2.md` §44 -- is what `:718` needs, and
that is a change to `irep2`, not to Solidity.

### 15.3 A fourth reader of a symbol's value

`frontends-to-irep2.md` §61 recorded three readers. There is a fourth, and it is the one
that makes attribute loss observable: `solidity_convert_constructor.cpp:499` reads
`symbol->get_value()` for a state variable and branches on `rhs.get("#zero_initializer")`
(`:503`, `:516`) and, through `convert_type_expr`, on `#sol_type`, `#sol_bytesn_size` and
`#sol_array_size`, including a full `irept` inequality between source and destination types.
`grep -c 'sol_type\|sol_bytesn_size\|sol_array_size' src/util/irep/migrate.cpp` returns **0**:
none of those attributes crosses the seam.

No program in the corpus shows a difference from that -- the GOTO is identical across 515 --
but the mechanism is real and it is the reason the remaining Solidity writes cannot be swept.
A write whose value reaches `:499` has to keep its `#sol_*` attributes, and `migrate_expr`
does not.

### 15.4 What this leaves

```
clang-c-frontend           1137     1224     1189       32     19
clang-cpp-frontend          631      683      669       15      3
solidity-frontend          1413     1625     1587       98     65
python-frontend            6528     7156     6964      108     54
jimple-frontend              97      118       96       10      3
total                      9806    10806    10505      263    144
```

Sixteen conversions across eight files, Phase 8's B-2\* at 65 and the total 144. Every one
of the sixteen is exercised -- 5 823 executions between them -- and both the GOTO and the
symbol table are identical for all 515 programs, so nothing can bite on them and nothing is
owed.

Three groups are deliberately left, each for a different reason: `:718`, which needs the
constant's spelling carried across the seam; the four writes taking
`gen_zero(get_complete_type(t, ns), true)`, which need the IREP2 `gen_zero(const type2tc &)`
overload and, at `:367-369`, a `zero_initializer` attribute that `type2t` has no place for;
and `solidity_convert_decl.cpp:615` with `solidity_convert_mapping.cpp:478`, `:559`, `:587`,
which no test in the corpus reaches at all. The last of those is the cheapest thing Phase 8
could fix next and the only one that is a test gap rather than a design question:
`mapping.cpp:478` needs `S s = m[k];` -- reading a struct by value out of a mapping -- which
no Solidity test in the tree does.

## 16. The synthesised low-level-call builtins, and a fourth name loss (2026-09-16)

`solidity_convert_call.cpp` held 19 of the 144 remaining B-2* writes -- the densest convertible
cluster left in any frontend -- in one repeated idiom. Each synthesised builtin builds a legacy
`code_typet t`, then a legacy `code_blockt func_body`, and writes both.

### 16.1 What converted, and the evidence it is inert

Nine of the 19 landed: the seven `set_type(t)` writes, and the two `set_value(gen_zero(...))` writes,
the latter hoisted into a typed local so the IREP2 `gen_zero(const type2tc &, bool)` overload
(`irep2_utils.h:335`) is selected rather than the legacy one, as a compile-time guarantee rather than
a comment. The `code_declt` operand on the following line deliberately keeps the legacy overload.

Over all 525 Solidity tests, base against change, `--goto-functions-only` **and**
`--symbol-table-only` are byte-identical -- **0 of 525 differ on either**, measured on a `DebugOpt`
build so `migrate_symbol_value`'s round-trip assertion (`migrate.cpp:490-504`) was live rather than
compiled out. 512 of the 525 digests are distinct, so the comparison is not the all-equal vacuity of
§158. Both flags write to **stderr**, so the capture is `2>&1`; a `2>/dev/null` capture hashes one
version banner and reports a false zero.

### 16.1.1 What that measurement cannot see, and why the writes are still safe

`migrate_type` here is **not** lossless, and the digests are blind to the loss by construction. The
seven code types are assembled from `solidity_convert.cpp:107-118`, where the members carry comment
keys the seam does not:

```cpp
addr_t = unsignedbv_typet(160);  set_sol_type(addr_t, ADDRESS);
bool_t = bool_type();            set_sol_type(bool_t, BOOL);  bool_t.cpp_type("bool");
```

`migrate_type` returns a bare `bool_type2tc()` / `unsignedbv_type2tc(w, cmt_constant())`, so
`#sol_type` goes from the return type and every argument type, and `#cpp_type` from the return type.
The dumps cannot register it: `show_symbol_table_plain` prints through `p->from_type` / `p->from_expr`
(`show_symbol_table.cpp:21-29`), the language pretty-printer. Only `irept::pretty` prints named_sub
and comments, and neither `-only` flag takes that path. So a byte-identical dump is *not* evidence of
attribute preservation -- it is evidence about what the printer chose to print. §15.3 said this of a
value write; it is equally true of a type write.

The blindness is not limited to comment keys. Clearing the argument identifiers on `$call#0` before
the store leaves 525/525 green **and** both dumps byte-identical, because `from_type` renders
`_Bool (Base *, unsigned _ExtInt(160))` -- argument types, no `#identifier`, no `#base_name`. Two
reasons it still cannot slip through: `migrate_type_back` does carry both (`migrate.cpp:3162-3173`),
and the carry has an in-tree gate. `migrate_symbol_type` asserts
`migrate_type(migrate_type_back(result)) == result` (`migrate.cpp:477`) on every symbol type the
pipeline reads, reached from `goto_convert_functions.cpp:1819`, so `ctest -L esbmc-solidity` on an
asserts build is what actually pins the seven type writes. Deleting the `argument_names` carry does
not merely fail a test -- `python2goto` aborts and the build stops.

That gate, not the dumps, is the losslessness evidence to cite. What the dumps establish is the
weaker and still useful claim that nothing a language printer reports has moved.

The remaining risk is the attributes no gate covers, and there a **reader** argument closes it. Of the 49
`get_sol_type` sites, only two read a code type's return type
(`solidity_convert_modifier.cpp:85`, `:191`), and both read a local `code_typet`, never
`symbol.get_type()`. The six unconditional builtins' call expressions take the member `bool_t`
directly (`solidity_convert_mapping.cpp:617-643`), not the symbol's type. The single read-back is

```cpp
// solidity_convert_call.cpp:2266
call.type() = to_code_type(helper_sym->get_type()).return_type();
```

and that type is never queried for `#sol_type` again -- it goes straight into
`convert_expression_to_code`. `#cpp_type` has four readers, none on a Solidity path. Argument
`#identifier` and `#base_name` *are* carried (`migrate.cpp:349-356`, `:3162-3173`), as is `ellipsis`.

One asymmetry this leaves, recorded rather than fixed: at the two `gen_zero` sites
`get_default_symbol` stores `rt` through the legacy setter, so `rs.get_type()` keeps `#sol_type` while
`rs.get_value()->type` does not. Nothing compares the two for these locals -- they are
`lvalue`/`file_local` and the initialiser rides the `code_declt`, not the symbol value.

### 16.1.2 The two value writes are verification-inert, and what pins them instead

Replacing the stored zero at `$dl_hret$` with a free symbol leaves all twelve `delegate_shadow` tests
passing; only the dump moves, `Value.......: 0` to the mutant. The GOTO takes the value from the
`code_declt` operand on the following line, so **no regression test can bite these two**, and asking
for a `SUCCESSFUL`/`FAILED` pair over them is asking for something that does not exist.

What pins them is a unit test, because the obligation is a pure-function property: the conversion
swapped `gen_zero(const typet &, bool)` for `gen_zero(const type2tc &, bool)`, and those are not the
same function. `unit/util/migrate.test.cpp` asserts they agree for every type reachable here --
`uint256`, `address` (`unsignedbv(160)` with `#sol_type`), `bool`, a signed integer, and Solidity's
`string`, which is a `pointer`. Mutation-checked: returning one instead of zero from the IREP2
bitvector arm fails it.

They diverge outside that set, which is why the test enumerates rather than generalises. On an
unhandled kind legacy returns a nil `exprt` (`expr_util.cpp:90-91`) where IREP2 logs and `abort()`s
(`irep2_utils.cpp:149-153`); and IREP2's array-of branch recurses with the default `false`
(`irep2_utils.cpp:97-98`) where legacy propagates the flag, so a nested array would trip
`assert(is_constant_int2t(arr_type.array_size))`. `get_complete_type` resolves symbol types before the
call, which closes the common route into the first, and no reachable Solidity return type takes the
second -- but the failure mode changed from a silent nil to an abort, so the boundary is worth naming.

Every converted line is exercised, proven by sweeping the corpus for the uniquely-named symbol each
builtin synthesises rather than by reading the code:

```
$call#0  $call#1  $transfer#0  $send#0  $staticcall#0  $delegatecall#0   523 of 525 tests each
$typed_call$                                                              4
$dl_ret$                                                                  3
$dl_hret$                                                                 1   (delegate_shadow_8)
```

The six unconditional builtins are synthesised for essentially every contract, which is why the
figure is 523 and not a feature-gated handful.

### 16.2 Why the other ten did not land

The eight `set_value(func_body)` and two `set_value(arg_exprs[i])` writes are also GOTO-neutral --
0 of 525 -- but they change **523 of 525 symbol-table dumps**, and the cause is a name the seam does
not carry. This is the fourth instance of that class, after `argument_base_names` (#7798),
`member_base_names` (§46) and `#cformat`:

```cpp
// util/expr/expr_util.cpp:239-245
exprt symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.get_type());
  tmp.identifier(symbol.id);
  tmp.name(symbol.name);      // <-- symbol2t has only `thename`
  return tmp;
}
```

`migrate_expr_back` rebuilds the identifier and not the `name`, so one symbol acquires two
inequivalent spellings: the legacy one inside an unconverted decl, and the `name`-less one inside a
converted body. `c_expr2stringt::get_shorthands` (`util/lang/c_expr2string.cpp:40-58`) collects
symbols into a `std::set<exprt>` and compares whole `exprt`s, so the two spellings do not dedupe,
their shared shorthand registers as a namespace collision, and the printer falls back to the full
mangled id:

```
base:   unsigned _ExtInt(160) old_sender=msg_sender;
after:  unsigned _ExtInt(160) sol:@C@AddmodOverflow@F@old_sender#1=msg_sender;
```

On this corpus it is cosmetic: the GOTO does not move, 0 of 525. The *field*, however, is not
cosmetic, and the clang-c side already knows it. `clang_c_adjust::do_special_functions`
(`clang_c_adjust_expr.cpp:1406`) dispatches every builtin lowering on
`to_symbol_expr(f_op).name()` -- the `name`, not the identifier -- so a callee that lost it stops
matching, and §90.2 records what that cost: an `assert` left as a plain `FUNCTION_CALL`. Two sites
already reconstruct the name by hand rather than live with that
(`clang_c_adjust_irep2.cpp:1597` and `:1647`).

### 16.3 What that leaves

Solidity's B-2* is 65 -> 56, the repo total 144 -> 135. The ten deferred writes are blocked on one
question, and it is not a per-site question: whether `migrate_expr_back`'s symbol arm should set
`name(get_pretty_name(id2string(thename)))`.

That is cheaper than it first looked. `get_pretty_name` (`util/symtab/pretty.h:9`) is a pure string
function -- no symbol-table lookup, and no new field, which matters because `symbol2t` is the
most-constructed node in the tool and must not grow. And the derivation is not speculative: it is
exactly what `clang_c_adjust_irep2` already does at the two sites above, by hand, because the name
was missing. Doing it once at the seam would retire those workarounds, the §90.2 class of defect, and
these ten writes together.

What it still needs is its own measurement, because setting `name` changes `irept::operator==` for
every back-migrated symbol expression -- which is the point (it is what removes the spurious
collision) and also the risk. That is a corpus-wide A/B across every frontend, not a rider on a
nine-site Solidity change.
