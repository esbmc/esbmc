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

| Suite | Tests | CORE | THOROUGH | KNOWNBUG |
|---|---:|---:|---:|---:|
| `esbmc-solidity` | 525 | 324 | 193 | 8 |

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
   `solidity_languaget::typecheck` and report the divergence count over the
   corpus. Measured baseline, no porting. **Done — §7.**
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
S.3, the converter's own 1 685 sites. §7.22's padding row is the one open
defect in the adjust and seam half; §7.23 has the corpus figures.

## 7. S.1 executed: the baseline, and it is one cause (2026-09-11)

The wiring mirrors `clang_cpp_language.cpp` — there is no C++ shadow mode, only
`clang-cpp-irep2-adjust-only`, so the gate is a single `if`/`else`. Default path
unchanged.

Before the wiring the flag was inert here: `abi_decode_1` run with it was
byte-identical to the same run without. After it, that test SIGSEGVs, which is
how the wiring is observed at all — see §7.3.

### 7.1 The measurement

Stride-8 sample of the corpus, each row run twice on one binary, flag off
against flag on:

| | rows |
|---|---:|
| measured | 65 |
| verdicts agree | **5** |
| verdicts diverge | 9 |
| crash under the flag | **51** |

A sample, not the whole corpus, and labelled as one: the machine this ran on
was under memory pressure heavy enough to have an earlier build OOM-killed, and
a 517-row sweep is 1 034 runs. The conclusion does not turn on the precision —
78 % of a stride sample crashing is not a figure a fuller run reverses into
health.

This is R1 arriving as a number rather than a prediction. Wiring the flag
exposes Solidity to every open Phase 7 divergence at once, which is exactly why
S.1 was specified as a measurement and not a flip.

### 7.2 The crashes are one site

Seven crashing rows sampled across the corpus — `abi_decode_1`, `bitwise_ops_2`,
`array_2`, `clearing_mapping_1`, `error_3`, `mapping_12`, `super_3` — symbolised
with `--segfault-handler` and `addr2line`. All seven share one top frame:

```
is_constant_bool2t(irep_container<expr2t> const&)   src/irep2/expr_kinds.inc:23
goto_convertt::optimize_guarded_gotos(goto_programt&)
                                     src/goto-programs/goto_convert.cpp:102
```

`optimize_guarded_gotos` tests `is_true(it_goto_y->guard)`, which is inlined,
and `is_constant_bool2t` dereferences the container. A GOTO instruction is
therefore reaching that pass with a **nil guard** where the legacy path leaves
a `true` one — so the defect is upstream of `goto_convert`, in what the IREP2
pass fails to fill in, not in the optimisation.

Naming the site is not naming the cause. Which expression is left nil, and by
which missing arm, is the next investigation; Phase 7's §3.3 found the same
shape (23 crashes, one site) and the cause was an unpopulated list the
converter leaves empty. That is a hypothesis here, not a finding.

### 7.3 What pins this, and what cannot

Nothing end-to-end. A flag-pinned Solidity test over one of the 5 agreeing rows
passes with the wiring in or out, so it pins nothing; a test pinning a diverging
or crashing row would be pinning the defect. The evidence for S.1 is the
measurement above, reproducible with the two commands in §1.1 plus the flag.

The instrument becomes available the moment §7.2's cause is fixed: a Solidity
row that today crashes, pinned for *producing a verdict at all*, is a real gate
— `^VERIFICATION SUCCESSFUL$` cannot match a SIGSEGV. That test belongs to the
change that fixes the crash, not to this one. #7717 shipped the C++ pass the
same way, with its census as the evidence.

### 7.4 The crash localised: the generated harness, and it is not a missing arm

`gdb -batch -ex run -ex bt` on the reduced input gives the frame `addr2line`
could not:

```
#0  make_not (expr=...)                        src/irep2/irep2_utils.cpp:8
#1  goto_convertt::optimize_guarded_gotos      goto_convert.cpp:101
#2  goto_convert_functionst::try_convert_body_native
#3  goto_convert_functionst::convert_function
```

So it is `make_not(it->guard)` at line 101, not the `is_true` on the line before
it, and `it` is a **conditional GOTO whose guard is nil**.

**It reduces to four lines.** `solc` is available at `/tmp/extest/solc`, so new
contracts can be generated rather than picked from the corpus:

```solidity
pragma solidity >=0.8.0;
contract C {
    uint x;
}
```

`--goto-functions-only` plus the flag crashes on that; without the flag it dumps
the program. No function, no statement, no expression of the user's is needed —
which places the guard in code the converter *generates*, and the legacy dump
names it: `_ESBMC_Main_C (sol:@C@C@F@_ESBMC_Main_C#)`, the per-contract harness,
whose first branch is `IF !return_value$_nondet_bool$1 THEN GOTO 2`. That is
exactly the `if(x) goto z; goto y; z:` shape `optimize_guarded_gotos` rewrites.

Two things this rules out.

**It is not the native body converter.** Frame #2 is `try_convert_body_native`,
but `--no-irep2-native-body` does not avoid the crash: the legacy
`goto_convert_rec` path runs `optimize_guarded_gotos` over the same sequence
and dies identically. So the malformed guard is in the adjusted body, not in
either converter.

**It is probably not a missing arm.** An A/B of `--symbol-table-only` over
`string_concat_1`, flag off against flag on, differs in ten lines, of which
eight are a temp-directory path and blank lines in one printed body. The one
structural difference is a dropped `#location` that was **present and empty**
on a constructor call — the irept tri-state where `find()` cannot distinguish
absent from nil, and a mutable `location()` creates a third state. An arm that
failed to run would not produce a symbol table this close.

That pointed at the seam rather than the arm table. §7.5 tests it, and the lead
does not survive.

Stated as a hypothesis, not a finding: the symbol-table closeness makes a seam
loss the better explanation, but nothing here has yet shown that *this* loss is
what empties the guard.

### 7.5 The location lead is weaker than §7.4 read it, and gdb cannot close it

Run on the four-line contract instead of `string_concat_1`, the
`--symbol-table-only` A/B is tighter still: **one** structural line, a dropped
`* #location:` that was present and empty. But it sits on the *callee symbol of
a constructor call inside* `sol:@_ESBMC_Object_C#` — not on any function that
holds a conditional goto. Four generated functions do hold one:

| generated function | conditional gotos |
|---|---:|
| `sol:@C@C@F@$transfer#0` | 2 |
| `sol:@C@C@F@$send#0` | 2 |
| `_sol_init_` | 1 |
| `sol:@C@C@F@_ESBMC_Main_C#` | 1 |

So §7.4's "try the location restore first" was too strong a reading. The loss is
real and worth fixing on its own account, but it is in a different symbol from
the crash and nothing connects the two.

The harness shape is not the cause either. `_ESBMC_Main_C` is
`while (nondet_bool()) { _ESBMC_Nondet_Extcall_C(); }` after a `__ESBMC_HIDE:`
label, and the direct C analogue —

```c
_Bool nondet_bool(); void body();
int main(void) { HIDE:; while (nondet_bool()) { body(); } return 0; }
```

— converts cleanly under `--clang-c-irep2-adjust-only`, under
`--clang-cpp-irep2-adjust-only`, and on the default path. A side-effect loop
condition reached through a `__ESBMC_HIDE` label is handled.

**Why this stops here.** Naming the function needs the symbol at frame 3, and
`gdb` reports `symbol = <optimized out>`; `dest.instructions` cannot be walked
either, because every accessor is inlined (`Cannot evaluate function -- may be
inlined`). A `-O2 -DNDEBUG` build will not give up that name. The next step is
the technique the earlier phases used for exactly this: a temporary `fprintf`
in `convert_function` printing `symbol.id` before `optimize_guarded_gotos`,
which needs a rebuild. Four candidates and a two-second reproducer make that a
short run once a build is available.

What is settled: the guard is nil in a converter-generated body, in one of four
named functions, and neither body converter, nor the loop shape, nor — on
present evidence — the location seam explains it. §7.6 closes it, and overturns
the "probably not a missing arm" reading above.

### 7.6 The cause: an unported arm, and the A/B that hid it

A temporary `fprintf` in `convert_function` names the function in one run:
`sol:@C@C@F@_ESBMC_Main_C#`, the 207th of 207 conversions, with the SIGSEGV
immediately after it. A second `fprintf` dumping `symbol.get_value().pretty()`
for that symbol gives the two trees the printed C form could not. The `while`
condition is a `sideeffect` function call, and legacy against flag-on reads:

| | legacy | `--clang-cpp-irep2-adjust-only` |
|---|---|---|
| the call's type | `bool` (`#cpp_type: bool`) | **empty**, keeping `#sol_type: BOOL` |
| the callee symbol's type | `code`, with `arguments` and `return_type: bool` | **empty** |

An expression with no type is what `goto_convert` turns into a nil guard, and
`make_not` then dereferences it.

Both losses have one cause, in
`clang_c_adjust::adjust_side_effect_function_call` (`clang_c_adjust_expr.cpp`):
when the callee resolves to a context symbol it replaces `f_op` with
`symbol_expr(symbol)` — restoring the callee's `code` type from the symbol
table — and then calls `align_se_function_call_return_type(f_op, expr)`, which
sets the call's type to the callee's `return_type`. `clang_cpp_adjust`
overrides that helper to skip constructors. Neither the callee replacement nor
the alignment is ported: `grep -n 'align_se\|return_type'` over both IREP2
adjust passes returns nothing.

So the Solidity converter emits the call with an incomplete type carrying only
`#sol_type: BOOL`, legacy repairs it, and the IREP2 pass leaves it as it found
it. It is a missing arm after all, and a **C** one — so porting it serves the
C++ frontend too.

**The instrument was the mistake, not the reasoning.** §7.5 concluded "probably
not a missing arm" from a `--symbol-table-only` A/B that differed in one line.
That dump renders values as C source, and `expr2c` prints a call with an empty
type exactly as it prints a typed one: `nondet_bool()`. The tree differed all
along; the printer flattened it. Any future A/B over adjusted bodies wants
`pretty()`, not the C rendering — the same shape of error as measuring a decline
census from a goto dump instead of a verdict.

Ported in §7.7, which measures what it moves.

### 7.7 One arm, and it moves 50 of the 51 crashes (2026-09-12)

`adjust_call_signature` rebuilds the callee from the symbol table when the
converter left its type incomplete, and then calls a new
`align_call_return_type` hook — empty in `clang_c_adjust_irep2`, as
`clang_c_adjust::align_se_function_call_return_type` is empty for C, and
overridden in `clang_cpp_adjust_irep2` to take the callee's `return_type` and
skip constructors. The row goes before `adjust_call_arguments`, whose parameter
types come from the callee type this repairs.

Same stride-8 sample, same binary discipline, flag off against flag on:

| | before | after |
|---|---:|---:|
| verdicts agree | 5 | **56** |
| crash | 50 | **0** |
| neither | 9 | 9 |

Those are the corrected figures; §7.9 says what was wrong with the first set.

**The row has to be in two tables.** `clang_cpp_adjust_irep2` substitutes its
own arm table rather than adding to the C one (§3.1 of the clang-cpp scope
doc). With the row in the C table alone the arm compiles, links, and never
dispatches: all four reduced contracts still crashed. That A/B is the row's
mutation evidence, and it is worth stating because a null result there invites
discarding a correct diagnosis.

**The residue is one cause, not nine.** Every non-agreeing row now ends in
`ERROR: migrate expr failed` — `constructor_4`, `enum_2`,
`function_overload_2_fail`, `import_2`, `inheritance_1`, `inheritance_8`,
`return_6`, `send_ether_via_creation_1`, `try_catch_1`. They looked like
timeouts in the sweep (`on=[]`), and they are not: one re-run with a 300 s
budget exits in 0.6 s with that error. The row first reported as a remaining
crash, `github_6759_02`, was never one — §7.9.

`regression/esbmc-solidity/irep2_only_call_return_type{,_fail}` pin it,
generated with `solc --ast-compact-json` so the checked-in `.solast` matches
the corpus convention, and bounded with `--unwind 1 --no-unwinding-assertions`
because the harness runs past 200 s unbounded. Both halves flip to a SIGSEGV
when the arm is disabled, so neither verdict regex is satisfiable without it.
Default path unchanged: `esbmc-solidity` is 525 of 527 with the same two
`KNOWNBUG` rows that already passed, and `irep2_only` is 99 of 99.

What this does not claim: any gain for the C frontend. The callee refresh is in
the C pass because that is where legacy has it, but clang's converter does not
leave an incomplete callee type, and the alignment hook is empty for C by
design. The 97 pre-existing C rows passing is consistent with the arm being
inert there, not evidence of a C-side improvement.

### 7.8 The residue is Phase 7's cpp_new size defect, already fixed in #7726

The nine `migrate expr failed` rows are **not** this branch's doing: the
diverging set is identical before and after §7.7's arm (`comm -13` over the two
sweeps is empty). They are a separate defect that closing the crashes merely
uncovered.

`gdb -batch -ex 'catch throw'` against the current build — line numbers shift
between builds, which is why a probe placed from an older backtrace never fired
— gives the chain:

```
code_block operand loop          src/util/irep/migrate.cpp:2568
  -> sideeffect_assign, lhs      src/util/irep/migrate.cpp:2093
    -> cpp_new size operand      src/util/irep/migrate.cpp:2125   throws
```

So it is a `sideeffect_assign` whose lhs is a `cpp_new` whose size is an **empty
irept**. The site reads

```cpp
const exprt &sz = expr.cmt_size().is_not_nil() ? … cmt_size() : … size_irep();
migrate_expr(sz, thesize);
```

and `is_not_nil()` is true for a *present-but-empty* irept, so the empty `#size`
is selected and handed to a `migrate_expr` that has no handler for it.

That is the defect PR **#7726** already fixes, in commit `d28be5a7e2`, with a
tri-state-aware test:

```cpp
const auto carries_size = [](const irept &i) {
  return !i.id().empty() && !i.is_nil();
};
```

All four stacked Phase 7 branches carry `cpp_new_size`; master does not. So
Phase 8's residue and Phase 7's `cpp_new` size fix are one defect, and this
corpus is a second, independent corpus for it — worth recording on that PR
rather than fixing twice here.

Two corrections this section makes to §7.5-§7.7's reading. The empty operand is
the `cpp_new` **size**, not a sideeffect's `op0`; and the five-kind exclusion
list at migrate.cpp:2116 is a red herring, since `cpp_new` is *in* it. The
one-line `&& expr.op0().is_not_nil()` guard that suggested itself would have
papered over a defect that already has a correct fix in review.

What is left after this: confirming the link above by running this corpus on a
build of #7726, which needs that branch built rather than argued from the diff.

### 7.9 The crash count was wrong, and the instrument was mine (2026-09-12)

`github_6759_02` never crashed. Its `test.desc` runs `--goto-functions-only`,
and the GOTO dump contains the literal string `uncaught exception` from ESBMC's
own exception machinery; the sweep classified a row as a crash with an
unanchored `grep 'SIGSEGV\|uncaught exception'` over the whole output, so the
dump matched itself. Re-running that row under the flag, with and without
`--no-irep2-native-body`, produces no crash at all.

So the figures are 50 real crashes before the arm and **0** after, not 51 and
1, and the same row inflated both ends. The classifier now anchors on the
diagnostic lines:

```sh
grep -qE '^ESBMC caught SIGSEGV|^ERROR: uncaught exception'
```

This is the second time in this section a clean-looking measurement was the
instrument's fault rather than the subject's — §7.6 was a `--symbol-table-only`
A/B that renders a typeless call identically to a typed one. Both belong in §4's
gates: a census must state what it greps for, and a sweep over program *output*
must anchor its patterns, because a dump quotes the verifier's own diagnostics
back at it.

### 7.10 The whole corpus, and the residue is three named buckets (2026-09-12)

The stride sample was a stand-in; this is the corpus. 509 rows measured — 525
directories less 8 `KNOWNBUG` and 8 with no source file for the flags line to
name — each run twice on one binary:

| bucket | rows |
|---|---:|
| verdicts agree | **441** |
| `migrate expr failed` | 63 |
| SIGSEGV | 3 |
| `cannot remove side effect (assign…)` | 2 |

**No row produces a wrong verdict.** Every residual row fails to produce one,
which is the failure mode to want: the hop-off declines loudly rather than
answering differently. That is worth stating plainly, because it is the
property the migration's gates exist to protect, and a 67-row divergence list
reads much worse than it is until the buckets are named.

Two figures moved while writing this up, both my instrument's fault rather than
the subject's. 441, not 439: two of the "divergences" were the pair this branch
itself adds, whose `test.desc` already pins the flag, so the sweep supplied it
a second time and ESBMC rejected the repeated option. The sweep now skips a row
that pins it, the way the Phase 7 reach probe already did. And the 63 is the
§7.8 bucket, still one cause, still #7726's.

**The 3 SIGSEGVs are new work**, and the stride sample missed all three:
`interface_7`, `struct_1`, `struct_2`. So is the 2-row `cannot remove side
effect (assign…)` bucket. Neither has been reduced yet; both are named here so
the next pass starts from a list rather than a sweep.

### 7.11 The 3 SIGSEGVs are one site, and it is in the solver (2026-09-12)

They do not reproduce from a reduced struct: a contract with a one-field
struct, its literal constructor, and a member read all convert and verify under
the flag. What the three rows share is not the construct but the *strategy* —
`--k-induction` on `interface_7` and `struct_1`, `--incremental-bmc` on
`struct_2` — and they crash only when symex actually runs:
`--goto-functions-only` on `struct_1` converts cleanly, and legacy with
`--k-induction` is fine.

`gdb -batch -ex run -ex bt` on two of the three gives the same top frame:

```
#0 smt_solver_baset::convert_assign   src/solvers/smt/smt_solver.cpp:366
#1 smt_convt::convert_assign          src/solvers/smt/smt_conv.cpp:80
#2 symex_target_equationt             src/goto-symex/equation/symex_target_equation.cpp:139
```

Line 366 is `side2->assign(this, side1)`, so an SSA assignment reaches the
encoder with sides it cannot assign across — the shape a sort or structure
mismatch takes, and the same shape as the mixed-width `ieee_fma` the C work
declined in §138.2 of the clang-c doc rather than hand to the solver.

So this bucket is **not** a missing adjust arm producing a nil: the body
converts. It is a type the pass leaves inconsistent, surfacing only once an
equation is built. Naming the mismatched assignment needs the two sides printed
at that frame, which `-O2` will not give up — the same wall as §7.5 — so it
wants the `fprintf` treatment next, not another A/B.

§7.12 answers which assignment, and refutes the sort-mismatch reading.

### 7.12 Not a sort mismatch: a nested member read the solver cannot project

The mismatch reading was wrong, and the probe that refuted it was written to be
able to. A `fprintf` at `smt_solver.cpp`'s assign site, firing only when
`eq.side_1->type != eq.side_2->type`, prints **nothing** before the crash: the
two sides' types are equal, so the assignment is not ill-sorted at the top
level.

What `gdb` does give up, once the fields are read directly rather than through
accessors it refuses to call:

```
info locals            side1 = 0xa54c2f0        side2 = 0x51
p eq.side_1.ptr_->expr_id                       expr2t::symbol_id
p eq.side_2.ptr_->expr_id                       expr2t::member_id
p eq.side_2.ptr_->type.ptr_->type_id            type2t::unsignedbv_id
p ((member2t*)eq.side_2.ptr_)->source_value.ptr_->expr_id
                                                expr2t::member_id
p …->source_value.ptr_->type.ptr_->type_id      type2t::struct_id
```

`side2` printed as `0x51`, and the crash is the virtual call on it rather than
the assignment itself. Treat the *value* with suspicion: these are `-O2`
locals, where `info locals` can show a stale register. What the frame supports
is that the RHS AST is unusable at the call, not that it is specifically `0x51`
— §7.14.

The RHS is a **nested member read**: `member(member(…, struct), unsignedbv)`,
which in `struct_1` is `this->book.book_id`. The outer component's name is an
`irep_idt` the debugger can only show as a pool index, so it is not resolved
here.

That makes the hypothesis worth testing next: the struct type reached through
the inner member differs between the two paths, so projecting the outer
component by name finds nothing and the flattener returns a bad AST. The pass
has two places that could do it — `adjust_struct` pads struct *literals*,
`pad_type_symbol` pads type *symbols* — and a type padded on one path and not
the other would behave exactly like this. Comparing the two struct types at
that site is the next instrument; it is not established yet, and the earlier
`--symbol-table-only` A/B cannot settle it, for the reason §7.6 records.
### 7.14 Both probes refute their hypothesis, and one earlier reading was over-read

The `fprintf` at `get_member_name_field`'s fall-off — printing the wanted name
and the names present whenever the scan runs off the end — **never fires** on
any of the three rows. So the name is always found, `idx` is in range, and
§7.13's out-of-range projection, though real as a mechanism, is not what these
rows hit.

That is two probes in a row that refuted the hypothesis they were built for,
which is the intended use: each was placed so that silence was an answer rather
than an absence of one. Where it leaves the bucket:

| claim | status |
|---|---|
| three rows, one frame — `side2->assign(this, side1)` | measured |
| crash needs symex — `--goto-functions-only` is clean, legacy is clean | measured |
| the assignment's two sides have equal types | measured (silent probe) |
| RHS is `member(member(…, struct), unsignedbv)` | measured |
| an ill-sorted assignment | **refuted** |
| a member name missing from its struct type | **refuted** |
| `convert_ast` returned the value `0x51` | **over-read** — an `-O2` local |

The retraction matters for the next step: with the lookup exonerated, the RHS
AST may be null rather than a stray non-pointer, and those two suggest
different culprits. So the next instrument prints `src` and its AST kind
*inside* `convert_member`, which separates "`project` misbehaves on a valid AST
and a valid index" from "the inner member conversion already failed and the
outer call inherited it".
### 7.15 Root cause: a padded struct type against an unpadded tuple (2026-09-12)

Printing `project`'s *result* as well as its input settles it. Both member
projections on the crashing path, with the source AST pointer, its sort kind,
the index, and what came back:

```
XPROJ src=0x2dad4d50 sort=6 idx=1 srckind=5    -> res=0x2dc28e10
XPROJ src=0x2dc28e10 sort=6 idx=3 srckind=58   -> res=0x51
```

The inner projection is fine. The outer one takes that AST and asks for field
**index 3**, and gets `0x51` back.

Read against §7.14, which established that `get_member_name_field` *finds* the
name: `idx = 3` is a valid position in the member-name list of
`member.source_value->type`, so that type has **at least four** members. `Book`
in `struct_1` declares three — `title`, `author`, `book_id` — so the type
carries a synthetic pad. And `project(3)` returning garbage means the AST's
tuple has **at most three** fields.

So the expression's struct type and the AST's tuple sort disagree on member
count: **the type is padded, the AST was built unpadded.** That is §7.11's
padding hypothesis with both halves measured instead of assumed, and it
explains why the two probes before it were silent — neither the assignment's
types nor the name lookup is wrong. The disagreement is between a type and an
AST built from a different version of the same type.

It also reverses §7.14's retraction of `0x51`. That value is real: it is
`project`'s return, printed by the probe, not an `-O2` local. The caution was
right in kind and unnecessary in fact, and what resolved it was printing the
*result* beside the input — a probe that reports only its inputs cannot tell
"went in bad" from "came out bad".

Two candidates for which side is stale, and this does not yet choose between
them: `pad_type_symbol`, which pads type symbols under `sole_adjuster`, and
whatever built the tuple sort — a sort cached from the symbol's type before
padding would behave exactly like this. Choosing needs the member counts of
`member.source_value->type` and of `src->sort` printed side by side, which is
one more line in the same probe.

Also worth separating out, as §7.13 noted for a different reason: `project`
taking an index it cannot bounds-check, from a lookup whose only guard is an
assert compiled out of release builds, turns any such disagreement into
undefined behaviour rather than a diagnosable failure.
### 7.16 Both counts measured: one struct type, padded in one place and not another

The inference in §7.15 was indirect — a found index on one side, a bad pointer
on the other — so the probe was extended to print both counts outright, firing
only when the index is out of range for the AST:

```
XCNT OOR idx=3 type_members=4 ast_members=3
```

The expression's struct type has **four** members: `Book`'s three declared
(`title`, `author`, `book_id`) plus a synthetic pad. The AST's tuple sort has
**three**. So the projection is out of range, and the type is padded while the
AST is not — as inferred, now read directly.

The locating detail is *which* AST. The inner projection returned a valid
struct AST with three fields, and that AST is the outer member's source. Its
sort was not built from the outer member's `source_value->type`, which has
four; a projected field's sort comes from its **parent tuple's** declared field
sorts. So the enclosing struct declares its `Book` field with an *unpadded*
`Book`, while the member expression reading that field carries the *padded*
one. Two `Book` types coexist, and they disagree.

That is the defect, stated as narrowly as the evidence allows: the pass pads
some occurrences of a struct type and not others, so a field's declared type
inside an enclosing struct disagrees with the type on expressions that read it.
`pad_type_symbol` pads *type symbols* under `sole_adjuster` and `adjust_struct`
pads struct *literals*; a `Book` inlined into another struct's member list is
reached by neither. Which of those two should also cover it is a
padding-ownership question larger than this branch, and it is not answered
here.

Three probes were needed and two of them refuted their own hypothesis, which is
the shape to want: each printed something whose *absence* was also informative.
The one that finally landed differs from its predecessors only in printing both
sides of the comparison rather than one.
### 7.17 The last unexplained bucket, closed by two docstrings (2026-09-12)

Both `cannot remove side effect` rows abort on `(assign_shr)`, and the cause is
written down in the tree twice over. `clang_c_adjust_expr.cpp`:

> The C converter now picks the kind (§76); Solidity still emits the untyped
> `assign_shr`, so the rewrite below stays for it.

so legacy resolves `>>=` to `assign_lshr` or `assign_ashr` by the target's
signedness, and `goto_sideeffects.cpp` handles only those two. The IREP2 arm
returns early on every shift spelling, and its helper says why that looked safe:

> The shift spellings clang_c_adjust returns early on: it promotes only the right
> operand there, which **the corpus shows** is already the migrated shape.

That was measured on the **C** corpus, where the converter resolves the kind.
Solidity emits a shape the C corpus does not contain, so the early return was a
sound conclusion from an incomplete population — which is §2's argument for
wiring this frontend in as a second corpus, paying for itself.

`adjust_compound_assignment` now performs the rewrite before that early return,
guarded as legacy guards it: `assign_shr`, a numeric right operand, and a
signed or unsigned target.

| row | before | after |
|---|---|---|
| `compound_assign_1` | `cannot remove side effect` | **agrees** |
| `op_binary_3` | `cannot remove side effect` | `migrate expr failed` |

So the bucket is empty and the **total residue is unchanged**: one row closes,
one advances past its first blocker onto §7.8's, which already has an owner in
#7726. Worth stating that way round — "two rows fixed" would be wrong.

`irep2_only_shift_assign_kind{,_fail}` pin it, and both halves flip to the
abort when the rewrite is disabled: an abort prints no verdict line, so neither
`^VERIFICATION SUCCESSFUL$` nor `^VERIFICATION FAILED$` is satisfiable without
it. `irep2_only` is 101 of 101, and the default path is unchanged at 527 of 529
with the same two `KNOWNBUG` rows that already passed.

Phase 8's residue is now 64 rows owned by #7726 and 3 by §7.16's padding
disagreement, with nothing unexplained, against 442 of 509 agreeing and no row
anywhere answering differently.
### 7.18 The padding fix that did not work, and what it rules out (2026-09-12)

§7.16 offered two candidates for the stale side. The first was tried and is
wrong.

`pad_type_symbol` pads only the top-level type of each type symbol, while
`clang_c_adjust::adjust_type` recurses — it walks each component before padding
the enclosing type, so a struct inlined into another's member list is padded
too. That looked like the whole story, so the IREP2 side was made to recurse
the same way (array subtypes, then components, then the enclosing type, leaning
on `add_padding`'s idempotence, which `adjust_type` asserts). All three rows
still SIGSEGV, so the change was reverted rather than parked: a change that
fixes nothing measurable should not ship.

What that rules out is useful. The short type is **not** a nested type symbol
the pass failed to reach, and the tree says where it does come from —
`adjust_struct`'s own comment:

> The literal's own type is an inline copy the converter recorded before
> `add_padding` ran, so `ns.follow` leaves it short the synthetic members.

So the 3-member `Book` is an inline copy carried **on an expression**, while
the 4-member one is the same struct resolved through the padded symbol table.
Two versions of one type in one tree, which is what §7.16 measured, but the
stale copy is on an expression and no amount of padding type *symbols* reaches
it.

That reframes the row: it is not a padding-ownership question but the seam
question §2.1 raised from the other end — which types on expressions are inline
snapshots and which resolve through the table. `adjust_struct` repairs that for
struct *literals* by padding their operands; nothing repairs it for a struct
type reached through a member read. The candidate fix is therefore to resolve
such a type through the table at the point the member is adjusted, in the way
`adjust_call_signature` already does for a callee's `code` type (§7.7) — the
precedent is on this branch, and it is the same class of repair.
### 7.19 S.2 done: the census names the symbol, and the bucket is not uniform

`migrate_census` was a `static` in `clang_cpp_language.cpp`. It walks a
`contextt` and is frontend-agnostic, so it moves into
`src/util/irep/migrate.{h,cpp}` beside the migration it measures, the C++
frontend's copy is deleted, and both frontends call one definition. Two copies
that can drift is the arm-table hazard (§7.7) applied to a helper.

**It answers a different question depending on the other flag, and that is easy
to misreport.** On `enum_2`, census alone:

```
IREP2 migrate census: 807 symbols, 357 values migrated, 8 type kinds, 0 failures
```

and census with `--clang-cpp-irep2-adjust-only`:

```
ERROR: IREP2 migrate census: migrate expr failed:  on symbol
       sol:@C@FreshJuiceSize@F@FreshJuiceSize#
IREP2 migrate census: 807 symbols, 356 values migrated, 8 type kinds, 1 failures
```

The first prices the **converter's** output, which is clean. The second prices
the **round trip** through the IREP2 pass. Quoting the first as "migration is
clean" would be wrong in the way §7.9 and §7.10 were wrong: a measurement that
runs, produces a plausible number, and answers a different question than the
one asked.

**What it says about the residue.** Over the 65 diverging rows it can run (two
pin the flag themselves):

| failures reported | rows |
|---|---:|
| 1 | 58 |
| 2 | 3 |
| 3 | 1 |
| 0 | 1 (`compound_assign_1`, closed by §7.17) |
| no census line | 2 (`bitwise_ops_1`, `op_binary_1`) |

and the failing symbols are **29 constructors and 33 methods** — so the bucket
is not constructor-shaped, as the one reduced row suggested, and some rows
carry more than one failure.

What this does *not* establish: that all 62 share §7.8's `cpp_new` size cause.
The error text is identical everywhere — `migrate expr failed:` with an empty
id, which is what an empty irept prints — and the one row traced by backtrace
was that cause, but identical text is not identical cause. #7726 merging is the
cheap test: re-run this census on that branch and the bucket either empties or
splits.

**Two flaws in the census harness, found before its numbers were used.** The
first run reported "46 rows: 1 failure, 15 rows: 0" and both figures were junk:
the symbol grep assumed no spaces between "census:" and "on symbol", where the
text is `migrate expr failed: `, so every symbol read as absent; and the script
passed each row's source file but dropped its **flags line**, losing
`--contract` and converting a different contract than the row verifies. That is
the fourth harness error in this section, against a comparable number of real
defects. The standing correction: do not quote a sweep figure without
re-reading what the sweep passed and what it grepped.
### 7.20 The #7726 link, tested rather than argued (2026-09-12)

§7.19 refused to claim that all 63 `migrate expr failed` rows shared §7.8's
cause: the error text is identical everywhere because an empty irept prints as
nothing, and identical text is not identical cause. So it was tested.

Cherry-picking `d28be5a7e2` conflicts — S.2 touches the same file — and that
commit also moves the typecast arms and the pre-dispatch forms, none of which
this question needs. So only the tri-state size selection was hand-applied, on
a scratch branch, marked in-source as an experiment:

```cpp
const auto carries_size = [](const irept &i) {
  return !i.id().empty() && !i.is_nil();
};
```

Census over the same 65 rows, before and after:

| failures reported | before | with the guard |
|---|---:|---:|
| 1 | 58 | 0 |
| 2 | 3 | 0 |
| 3 | 1 | 0 |
| 0 | 1 | **63** |
| no census line | 2 | 2 |

Every row with a migrate failure clears, across both symbol shapes — 29
constructors and 33 methods — so the bucket really is one cause, and #7726
closes all of it. The generalisation held; the point is that it was checked
instead of assumed.

A verdict sweep with the guard applied reports **64 agree, 0 diverge, 1 crash**
over its sample, the crash being `struct_2` from §7.16. Read that against the
56/9/0 of §7.9 with care: the two stride samples do not have identical
membership, because the skip lists differ between the two versions of the sweep
(one skipped no `KNOWNBUG` row, the other one). The robust statement is that
divergences in the sample fall to zero and the only residual is the padding
crash.

This is #7726's fix, not work owed here. The guard stays on the scratch branch;
what is owed is a note on that PR that this corpus exercises it across 63 rows.

One measurement detail worth stating, since it makes two numbers non-comparable
if left out: `enum_2` cannot answer at all without `--goto-functions-only` —
the full run exceeds 200 s — so the census and the verdict sweep measure that
row at different depths. Fine for counting migrate failures, which precede
symex; not fine to quote side by side unremarked.
### 7.21 A fourth bucket, and why this one had to be fixed at the seam

The two rows S.2 could not census at all (§7.19) name themselves, unlike the
empty-irept bucket:

```
ERROR: migrate expr failed: shr
```

An id, not a blank. `migrate_expr` has no handler for a plain `shr`, and cannot
have one usefully: **IREP2 has no kind-less shift node** — `is_shift` covers
`shl2t`, `ashr2t`, `lshr2t`. `clang_c_adjust` resolves it before anything
migrates, by the left operand's signedness (`unsignedbv` -> `lshr`, `signedbv`
-> `ashr`).

This is the *expression* twin of §7.17's `assign_shr`: the Solidity converter
emits both `>>` and `>>=` without picking the kind, the C converter picks both,
and the
IREP2 side assumed resolution because the C corpus always had it. Three of Phase
8's four fixes are now that same defect class.

**The asymmetry is where each has to be repaired.** `assign_shr` survives
migration, so an adjust arm can rewrite it. `shr` cannot be migrated at all, so
no arm ever sees it and the seam is the only code on the flag path that runs
early enough. That is a more invasive place for a semantic decision, so the
helper says why rather than leaving a bare special case.

| row | before | after |
|---|---|---|
| `bitwise_ops_1` | `migrate expr failed: shr` | **agrees** |
| `op_binary_1` | `migrate expr failed: shr` | `migrate expr failed:` (§7.8's bucket) |

One row closes; one advances past `shr` onto the `cpp_new` size blocker that
§7.20 showed #7726 closes.

**The complexity gate rejected the first attempt, correctly.** Three branches
added inline took `migrate_expr` from 291 to 294, and the gate blocks any
increase on a function already over threshold. #7726's own commit message
describes the remedy — move arms out into named helpers — so the existing
`lshr` and `ashr` arms were extracted alongside the new `shr`: two branches
removed, one added, net decrease, gate clear. Pleading the case was not an
option, and should not have been the first instinct.

Because the extraction moved arms every frontend uses on the **default** path,
the regression net is wider than for a flag-gated arm: 864 of 864 unit tests,
103 of 103 `irep2_only`, both shift pairs re-run against the refactored build
rather than assumed, and the Solidity default path unchanged.

One thing to hand forward: `adjust_compound_assignment` now sits at CCN **15**,
exactly the `core` gate line, so the next branch added there fails the gate.
Extract before adding.
### 7.22 Why the padding row is not a one-arm port (2026-09-12)

§7.18 ruled out nested type symbols. The reading is sharper than that, and it
explains why the obvious repair cannot work. Both numbers come from the **same**
expression:

- `get_member_name_field(member.source_value->type, …)` finds **4** names, so that
  expression's type is the padded `Book`;
- `convert_ast(member.source_value)` yields a tuple of **3**, so the AST was built
  from a different version of that type.

The AST cannot come from the expression's own `->type`. The inner member's AST
is `src->project(idx)`, and a projected field's sort comes from the **parent
tuple's** declared field sorts — so the grandparent's struct type carries an
*unpadded* `Book` component while the child member's own type is the *padded*
one. The disagreement is within a single expression tree, between a
grandparent's component type and a child's type.

That kills the repair §7.18 proposed. Retyping the member, or its source, does
not touch the AST: the sort is already fixed by the grandparent. Nor does
padding type symbols harder, which §7.18 measured. The fix has to retype the
**base** of the member chain so the whole tuple chain is built from padded
types — a recursive type-normalisation over expressions, not an arm.

That is a design question about which types on expressions are inline snapshots
and which resolve through the table — §2.1's seam question, reached from the
other end — and it is not worth a speculative broad change for 3 rows of 509.
It is recorded here as the remaining Phase 8 defect, with the two dead ends
marked so the next attempt does not repeat them:

| attempt | outcome |
|---|---|
| pad nested type symbols recursively (`pad_type_tree`) | no change; the stale type is on an expression |
| resolve the member's own source type through the table | cannot work; the AST's sort predates it |
| retype the base of the member chain | untried, and the only one that reaches the sort |
### 7.23 The corpus after four fixes, and what is left (2026-09-12)

Full sweep, 531 directories, 507 measured — 8 `KNOWNBUG`, 6 that pin the flag
themselves (this branch's own tests, skipped by the rule §7.10 added), and the
rest lacking a source for their flags line to name:

| | earlier (§7.10) | now |
|---|---:|---:|
| verdicts agree | 441 | **441** |
| diverge | 67 | **63** |
| crash | 3 | 3 |
| measured | 509 | 507 |

Divergences fall by four: `bitwise_ops_1` and `compound_assign_1` agree, and
the two artefacts §7.10 found are now excluded rather than miscounted. The
agreement count is flat because the rows the other two fixes touched moved
*within* the residue rather than out of it — `op_binary_1` and `op_binary_3`
each advanced past their first blocker onto §7.8's.

**Nothing answers differently.** Of the 63 divergences, rows where both paths
reach a verdict and disagree: **zero**. Every residual row declines — loudly,
with an error — rather than returning a different answer. That is the property
the migration's gates exist to protect, and it has held through every sweep in
this section.

The residue is fully owned:

| bucket | rows | owner |
|---|---:|---|
| `migrate expr failed` | 63 | #7726, measured closing all 63 (§7.20) |
| padding disagreement | 3 | §7.22, open — `interface_7`, `struct_1`, `struct_2` |

So with #7726 merged this corpus reaches **504 of 507**, and the only open
defect is the type normalisation §7.22 describes. Phase 8 opened on a roadmap
entry that recorded the suite as unmeasurable (§1.1); it now has a measured
census, the hop-off wired, four mutation-checked fixes, and a residue with
named owners.

What is *not* done, and should not be read as done: S.3, the converter's 1 685
construction sites (§1.2), which is the phase's actual bulk. Everything above
is the adjust and seam half — the half Phase 7 shares — and it is what made the
converter's half measurable, not a substitute for it.
### 7.24 S.3 opened: the converter's output has no representational wall

Before ranking 1 685 sites, the prior question: does what `solidity_convertert`
builds fit in IREP2 at all? An IR that cannot hold a construct would gate the
whole phase behind a representation change, as W3 did for earlier ones.

The migrate census answers it when run **without** the hop-off flag — that
configuration prices the converter's own output rather than the round trip,
which is the distinction §7.19 had to learn the hard way. Stride-8 sample:

```
--- converter census, stride 8: 65 rows, 65 clean, 0 with failures ---
```

Every row migrates: `s.get_type2()` and `s.get_value2()` succeed on every
symbol the corpus produces. So **S.3 has no W3-style wall.** Its risk is volume
and behaviour preservation, not representation, and the porting order can be
chosen on cost rather than dictated by a blocker.

Two limits on that claim, stated rather than left implied. It is a stride-8
sample, not the whole corpus; and it measures only what the corpus exercises —
a converter path no test reaches is unmeasured either way, which is §4's own
rule about a census showing the thing under test executed.

**Pathfinder.** `solidity_convert_literals.cpp` is 159 lines with 15 sites
(§1.2): the smallest self-contained target, and the right place to establish
the pattern before `solidity_convert_call.cpp` (404) and
`solidity_convert_expr.cpp` (250), which are 39 % of the surface between them.
Phase 5 used jimple the same way — smallest surface first, as the kit's
pathfinder.

What S.3 inherits from §7's half: a corpus that agrees on 441 of 507 rows with
nothing answering differently, so a converter change that breaks something will
show as a *new* divergence against a known baseline rather than disappearing
into noise. That baseline is the deliverable of this section, more than any
single fix in it.
### 7.25 S.3's first blocker, before a line is ported: `#cformat`

§7.24 said the converter's output faces no representational wall. That is true
of *forward* migration, which is what the census measures, and it is not the
whole question. Any boundary that stays legacy needs the **reverse**, and the
reverse loses something.

The pathfinder's first function, `convert_integer_literal`
(`solidity_convert_literals.cpp`), builds

```cpp
the_val = constant_exprt(
  integer2binary(z_ext_value, bv_width(type)),
  integer2string(z_ext_value),
  type);
```

and that three-argument constructor records the decimal spelling as `#cformat`
(`util/irep/std_expr.h`):

```cpp
constant_exprt(const irep_idt &_value, const irep_idt &_cformat, const typet &_type)
{
  set("#cformat", _cformat);
  set_value(_value);
}
```

A native port would build `constant_int2tc(type, value)` instead, and
`migrate_expr_back` reconstructs a `constant_exprt` from **`value` alone**
(`migrate.cpp`, `constant_int_id`): it sets the binary string and nothing else.
`#cformat` does not survive the round trip.

So porting this one function, with its callers left legacy, either carries
`#cformat` across the seam or changes what ESBMC *prints* for every Solidity
integer literal — counterexamples and goto dumps include it. That is a
default-path output change of the kind the clang-c doc's §137.5 treats as
SV-COMP-relevant, not a silent internal refactor.

Three ways out, none free, and the choice belongs to S.3's design rather than
to this section:

| option | cost |
|---|---|
| carry `#cformat` through `migrate_expr_back` | a seam change affecting every frontend's integer literals |
| port the callers too, so no round trip happens | pushes the boundary into `solidity_convert_expr.cpp` (250 sites) |
| accept the printed-output change | needs an SV-COMP run and a sweep of tests that pin literal spellings |

This is the same shape as the spelling carriage `scope-c-spelling-carriage.md`
records, arrived at from a different direction. Worth knowing before 1 685
sites are ranked: the smallest file in the phase is blocked on a seam question,
so "mechanical" in §7.24 means *representable*, not *free*.
### 7.26 504 of 507, measured on the merged tree (2026-09-12)

#7717 and #7742 landed on master, and #7742's merge carried the stacked #7726
content with it — `cpp_new_size` is on master although #7726 is still open as a
PR. So the fix §7.20 measured on a scratch branch is now the shipped one.

That had to be re-measured rather than carried across: what landed is #7726's
fuller restructuring of the cast arms and pre-dispatch forms *around* the
guard, not the hand-applied `carries_size` lambda alone. Full sweep against the
merged tree:

```
--- 507 measured (stride 1), skipped 8 KNOWNBUG/FUTURE: agree=504 diverge=0 crash=3 ---
```

| | §7.10 | §7.23 | now |
|---|---:|---:|---:|
| verdicts agree | 441 | 441 | **504** |
| diverge | 67 | 63 | **0** |
| crash | 3 | 3 | 3 |

Zero divergences of any kind, and zero rows where both paths reach a verdict
and disagree — the property that has held through every sweep in this section.
The whole residue is §7.22's padding row: `interface_7`, `struct_1`,
`struct_2`.

So the adjust-and-seam half of Phase 8 is closed but for one defect, at 3 rows
in 507. The phase opened on a roadmap entry recording the suite as unmeasurable
(§1.1), and the first sweep after wiring the flag had 50 of 65 sampled rows
crashing (§7.1, §7.9).

What of that is this branch's: four mutation-checked fixes —
`adjust_call_signature` (§7.7), the `assign_shr` rewrite (§7.17), the shared
migrate census (§7.19), and the kind-less `shr` at the seam (§7.21) — plus the
flag wiring and the baseline itself. The 63-row bucket is #7726's, and this
corpus's contribution there was to measure that it closes all of them.

What is *not* done: S.3, the converter's 1 685 sites, now gated on §7.25's
`#cformat` question rather than on anything in this section.
### 7.27 The `#cformat` question, priced: it bites only above 2^63

§7.25 left three options without costing them. The printer decides which matter.

`c_expr2stringt::convert_constant` uses `#cformat` verbatim when present and
otherwise decodes `value`:

```cpp
const std::string &cformat = src.cformat().as_string();
if (cformat != "")
  dest = cformat;
```

and the bitvector fallback is not a plain decimal:

```cpp
BigInt llong_ub = BigInt::power2(config.ansi_c.long_long_int_width - 1);
...
else if (int_value >= llong_ub)
  dest = "0x" + integer2string(int_value, 16);
else
  dest = integer2string(int_value);
```

So dropping `#cformat`:

| literal | with `#cformat` | fallback | same? |
|---|---|---|---|
| below 2^63 | decimal | decimal | **yes** |
| at or above 2^63 | decimal | **hex** | no |

The Solidity converter always sets `#cformat` to the plain decimal rendering
(`integer2string`, after normalising scientific notation), which is exactly
what the fallback computes below 2^63. Above it, the fallback switches to hex
and the printed text changes — and `uint256` makes that an ordinary case, not a
corner: 14 contracts in this corpus use a ≥19-digit literal or
`type(uint256).max`.

**Blast radius, measured.** No `test.desc` in `esbmc-solidity` pins a ≥19-digit
decimal, so no current expectation breaks. The change would be invisible to the
suite and visible in counterexamples — which is the combination that makes it
worth stating rather than discovering later.

That re-prices §7.25's options:

| option | revised cost |
|---|---|
| carry `#cformat` across the seam | needs a spelling field on `constant_int2t`; a W3-shaped change |
| port the callers too | no round trip, no spelling question; pushes into 250 sites |
| accept the change | free below 2^63, changes counterexample text above it; no test pins it today |

The third is cheaper than §7.25 implied, and still not free: it silently alters
how large `uint256` values print. Given that the suite would not catch it, a
deliberate choice with a note in the PR beats a quiet one — and porting the
callers avoids the question entirely, which is the argument for taking
`solidity_convert_literals.cpp` and `solidity_convert_expr.cpp` together rather
than the smallest file alone.
### 7.28 §7.22 was wrong: every type agrees, the *sort* is stale (2026-09-12)

§7.22 said a grandparent's `Book` component was unpadded while the child
member's own type was padded. Measured, that is false. A probe walking the
member chain from the failing projection to its base, printing each link and
each base component:

```
XCHAIN d=0 kind=58 members=4        (the inner member, this->book)
XCHAIN base d=1 kind=5 members=10   (the base: a symbol, the contract struct)
XCHAIN   base component 0 (Book) is a struct with 4 members
XCHAIN   base component 1 (book) is a struct with 4 members
```

Kind 58 is `member`, kind 5 is `symbol`. So the base's `book` component **is**
the padded four-member `Book`, and so is the member expression's own type. At
the crash, every *type* in the chain agrees at four members — and `project(3)`
still returns garbage, so the AST's tuple has fewer.

The stale thing is therefore the **sort**, not a type. The tuple sort the AST
carries was not built from the types above; it is a sort created earlier —
while some expression still carried an inline pre-padding `Book`, the case
`adjust_struct`'s comment describes — and reused afterwards for the padded
type.

Three consequences:

- §7.22's diagnosis is retracted. Type-versus-type was the wrong frame; it is
  type-versus-sort.
- It explains why `pad_type_tree` (§7.18) changed nothing, and confirms that revert
  was right rather than lucky: the types were already correct, so padding them harder
  could not help.
- The repair moves out of the padding question entirely. Either no expression may
  carry an unpadded inline struct type — a seam/pass normalisation, §2.1's question
  again — or the sort cache must distinguish the two, which is a solver change.

What is measured: all types four members, the AST fewer, the base a symbol with
a ten-member contract struct. What is not: which earlier expression created the
stale sort. That is the next probe — the sort's creation site, not another type
dump.
### 7.29 The sort is a wrapper, so a three-member type really was converted

Where the stale sort comes from, narrowed by reading rather than probing.
`convert_sort` caches on the type itself:

```cpp
typedef std::unordered_map<type2tc, smt_sortt, type2_hash> smt_sort_cachet;
```

and the symbolic flattener's `mk_struct_sort` does not declare or name anything
— it wraps the type:

```cpp
return new smt_sort(SMT_SORT_STRUCT, type);
```

So a struct sort *carries* the type2tc it was built from, and
`get_tuple_type()` returns exactly that. §7.16 read three members out of it.
Since padded and unpadded `Book` are different `type2tc` values, they are
different cache keys — nothing merges them.

Therefore a **three-member `Book` was genuinely converted** at some point in
the same run, and the AST that reaches the failing projection carries the sort
made from it. That is consistent with §7.28 and rules out the tidier
explanations: not a name-keyed declaration reused across two shapes, and not a
cache collision.

What remains is to find the expression that carried it. The probe for that logs
every struct sort conversion with its member count and struct name, and reports
the first three-member `Book` — a creation-site question, which is why the type
dumps of §7.22 and §7.28 could not answer it. `adjust_struct`'s comment already
names the shape to expect: an inline copy the converter recorded before
`add_padding` ran.
### 7.30 Two `Book` sorts exist, and both are made after symex

The creation-site log, in order, with the pipeline stages it falls between:

```
Symex completed in: 0.005s (135 assignments)
XSORT struct addr_space_type members=2
XSORT struct pointer_struct members=2
XSORT struct struct Book members=4      <- padded, created first
XSORT struct struct Book members=3      <- unpadded, created second
XSORT struct struct BytesPool members=2
XSORT struct Base members=10
ESBMC caught SIGSEGV
```

Three facts, none of them previously established:

- A three-member `Book` sort really is created, so §7.29's deduction holds: some
  expression in the equation carries the unpadded type. It is not a cache artefact.
- Both `Book` sorts are created **after symex**, during SSA conversion — so the
  unpadded type survives the adjust pass, migration, goto conversion *and* symex
  before anything notices.
- The padded one is created first and the unpadded second, and `Base`'s sort is
  created last, immediately before the crash.

That last ordering is the puzzle the next probe has to resolve. The failing
projection reads `book` out of `Base`, whose `book` component is four members
(§7.28) — so projecting it should reach the four-member sort that already
exists, not the three-member one. Something in that projection is not using the
component type the tuple was built from.

So the question is no longer "is there a stale sort" but "which expression
carries the unpadded type, and why does the projection prefer it". The probe
for that prints the expression kind and struct name at each `convert_ast` of a
struct-typed node, not the sorts alone.
### 7.31 The proximate cause: the index and the container come from different types

`tuple_sym_smt_ast::project` reads its member list from the **sort's** type:

```cpp
const std::vector<type2tc> &members =
  struct_union_members(sort->get_tuple_type());
assert(idx < members.size() && "Out-of-bounds tuple element accessed");
const type2tc &restype = members[idx];
smt_sortt s = ctx->convert_sort(restype);
```

while `convert_member` computes that index from the **expression's** type:

```cpp
unsigned int idx = get_member_name_field(member.source_value->type, member.member);
smt_astt src = convert_ast(member.source_value);
return src->project(this, idx);
```

Two different types, one index. When they agree the code is correct; when they
do not, `members[idx]` is an out-of-bounds `std::vector` read, `restype` is
garbage, `convert_sort` is handed it, and the resulting pointer is the `0x51`
of §7.12. The `assert` that would have caught it is compiled out of
`RelWithDebInfo` — §7.13's observation, now with the exact index and container
named.

That is the *proximate* cause, and it is independent of which expression
carries the unpadded type: any disagreement between an expression's struct type
and the type its AST's sort was built from becomes undefined behaviour here
rather than a diagnosable failure. §7.16's measurement — `idx=3`,
`type_members=4`, `ast_members=3` — is exactly this shape.

Two repairs, and they are not alternatives:

1. **Make the mismatch loud.** Derive the index from the same type `project` indexes,
   or check the bound in release too. This is a solver change, it fixes a class of
   undefined behaviour rather than one Solidity row, and it would have turned five
   ticks of probing into one error message.
2. **Remove the mismatch.** Ensure no expression carries a struct type that disagrees
   with its AST's sort — §2.1's seam question, still open, and still the migration's
   own work.

The first is worth doing on its own account and belongs to whoever owns the
solver's tuple layer; the second is what closes these three rows. Recording
both, because the first is the reason this defect cost what it did to find.
### 7.32 The offending expression is a nested struct literal — and it is padded

With #7758's guard in place the failure is reportable, so a probe in
`convert_member` can print the expression at the mismatch rather than inferring
it. The source is

```
XSEAM member=book_id
SOURCE member
* source_value : constant_struct        <- a struct *literal*, not a symbol
```

so the chain is `member(member(<literal>, book), book_id)`, and the inner `book`
value is itself a `constant_struct`. That explains at last why padding *type
symbols* never helped (§7.18): the stale value is a literal, which
`adjust_struct` owns, not a type symbol.

**But the literal is not short.** Its type carries four member names —

```
member_names : 0: title  1: author  2: anon_pad#2  3: book_id
```

— and it carries four operands to match (`symbol`, `symbol`, `constant_int`,
`constant_int`). So `adjust_struct` did its job: type and value agree at four,
and `anon_pad#2` is in place. `get_member_name_field` returning 3 for `book_id`
is right.

Yet the AST built from that literal carries a **three**-member sort. So the
disagreement is not between a type and a value, and not between two types: it
is between a correctly padded literal and the AST constructed from it. That
moves the question to `constant_struct2t`'s AST construction or to an AST
reused for a differently-typed literal — and away from padding entirely, for
the third time.

Also worth recording: the guard changes the outcome from a SIGSEGV to

```
ERROR: Tuple field 3 out of range: … struct Book …
ERROR: SMT solver failed
VERIFICATION UNKNOWN
```

so these three rows reclassify from *crash* to *divergence* in the sweep, and
the tool degrades gracefully with the cause on screen.

One methodological note: `gdb -ex 'catch throw'` is not a shortcut to this
throw. It stops at the first throw in the run, which here is an unrelated
internal one in `type_byte_size`'s `size_bits_expr` reached through
`convert_addr_of`, caught and handled. Reading that backtrace as the fatal path
would have been a fourth wrong-site diagnosis.
### 7.33 The mechanism, from `tuple_create`: operands size the tuple, the type sizes the sort

`smt_tuple_node_flattener::tuple_create` builds a literal's AST like this:

```cpp
tuple_node_smt_ast *result = new tuple_node_smt_ast(
  *this, ctx, ctx->convert_sort(structdef->type), name);
result->elements.resize(structdef->get_num_sub_exprs());
```

The **sort** comes from the literal's `type`; the **element count** comes from
its `get_num_sub_exprs()`. Nothing checks that the two agree. So a struct
literal whose operands were not padded while its type was produces a tuple with
fewer elements than its own sort advertises — precisely the state §7.31's guard
reports, and the reason the index from `get_member_name_field` (computed off
the type) can exceed it.

That makes the defect a struct literal that `adjust_struct` did not pad. §7.32
measured one `Book` literal in the failing expression with four operands and a
four-name type — correctly padded. But the expression contains **two** `Book`
literals, and the failing tuple holds three elements, so the second is the
unpadded one. `adjust_struct` reached one and not the other.

That is the target: not padding in general, not the sort cache, not the seam —
one arm that pads the literal it is dispatched on and misses a sibling. Why it
misses the second is the next question, and the arm's own dispatch is where to
look: the table visits nodes during the walk, so a literal rebuilt by an
earlier arm may not be revisited.

A `tuple_create` that refuses to build a tuple whose element count disagrees
with its sort would have caught this at the source rather than at the
projection, and is worth considering alongside #7758's guard.
### 7.34 `adjust_struct` sees the inconsistency and declines it

The arm, in full:

```cpp
  std::vector<expr2tc> ops = to_constant_struct2t(expr).datatype_members;
  if (ops.size() == st.members.size())
    return;

  ops = pad_struct_operands(st, ops);
  // A residual mismatch is not this pass's to guess at: leave the literal as
  // it stands rather than build one the type cannot describe.
  if (ops.size() == st.members.size())
    expr = constant_struct2tc(padded, ops);
```

The failing literal has **three operands and a four-name type** (§7.32 read the
four names; §7.31's guard reports three elements). So `ops.size()` is 3,
`st.members.size()` is 4, `pad_struct_operands` does not bring it to 4, and the
arm takes its documented bail-out: it leaves the literal alone rather than
building one its type cannot describe.

That bail-out is defensible read locally and wrong read end-to-end. The literal
it declines to touch is *already* inconsistent — a four-member type over three
operands — and `tuple_create` then sizes the sort from the type and the
elements from the operands (§7.33), so the malformed value reaches the solver
and the projection runs off the end.

Note what this says about provenance: the arm does not create the
inconsistency. The literal arrives with a padded type and unpadded operands, so
the padding of its
*type* happened elsewhere — `migrate_type` resolving the inline copy through the
padded tag symbol is the candidate, and the arm's own comment says `ns.follow`
leaves such a type short, which is no longer what it observes.

Two repairs, and the first is small:

1. **Do not leave an inconsistent literal.** If the operands cannot be padded to the
   type, put the literal back on a type that matches what it has, rather than
   letting a four-member type sit over three operands. The invariant to hold is
   type-length equals operand-length, either way round.
2. **Find why `pad_struct_operands` declines this shape**, which is the real fix if
   the operands *should* be paddable.

Worth adding to #7758's reasoning: a `tuple_create` that refused to build a
tuple whose element count disagrees with its sort would have stopped this at
the source, one layer before the projection.
### 7.35 §7.34 corrected: the padding helper would fix it, so the arm never ran

`pad_struct_operands` is four lines:

```cpp
for (size_t i = 0; i < st.members.size(); i++)
  if (i <= ops.size() && is_padding_name(st.member_names[i].as_string()))
    ops.insert(ops.begin() + i, gen_zero(st.members[i]));
```

and `is_padding_name` matches `anon_pad#` (`util/irep/pad_names.h`), which the
observed member name `anon_pad#2` satisfies. So for the failing shape —
operands `[title, author, book_id]` against names `[title, author, anon_pad#2,
book_id]` — it inserts a zero at index 2 and returns four operands. The helper
handles this shape exactly.

So §7.34's reading is wrong: the arm does not reach its bail-out on this
literal, because had it run, `ops.size()` would have become 4 and the literal
would have been rebuilt on the padded type. **The arm never ran on the failing
literal at all.**

That fits what §7.32 measured — two `Book` literals in one expression, one with
four operands and one with three. The arm fixed the first and never visited the
second.

So the target is the **walk**, not the arm's logic: which `constant_struct2t`
nodes does the dispatcher reach, and which does it skip? §7.33 guessed at this
already ("a literal rebuilt by an earlier arm may not be revisited"), and it is
now the only candidate left standing. The instrument is a log in
`adjust_struct` of every literal it visits, with its operand and type counts,
against the two the failing expression contains.

Six readings of this defect, five of them wrong in some particular, and each
corrected by an instrument built so it could say so. What survives from all of
them: the failing value is a nested struct literal with three operands under a
four-name type; nothing downstream reconciles the two; and the arm that would
have fixed it did not see it.
### 7.36 `adjust_struct` never sees a `Book` literal at all

The arm, logging every literal it visits with its operand and type counts:

```
28 XLIT visit struct BytesStatic  ops=2 type_members=2 padded_members=2
12 XLIT visit struct BytesDynamic ops=5 type_members=5 padded_members=5
 2 XLIT visit struct BytesPool    ops=2 type_members=2 padded_members=2
```

Three struct literals, all from the operational models, every one already
consistent. **No `Book` literal is visited — not the padded one, not the short
one.** So §7.35 was right that the arm never ran on the failing literal, and
understated it: the arm never runs on any `Book` literal, so `adjust_struct` is
not where this is fixed and the walk is not skipping a sibling it otherwise
reaches.

That relocates the defect again, and this time away from the adjust pass
entirely. The failing expression lives in the **SSA equation**, after symex
(§7.30 timed the sorts as post-symex). The `Book` literals in it are therefore
not in the set the pass walks: they are built later — by symex propagating an
initialiser, or by a value position the walk does not descend into. Either way,
a pass that runs over symbol values before goto conversion cannot repair a
literal that does not exist until symex has run.

So the open question is now provenance: **what constructs a three-operand
`Book` literal under a four-member type, after the adjust pass has finished?**
The candidates are the contract's initial-value construction in the Solidity
converter and symex's own struct propagation, and telling them apart wants the
literal's origin, not another count.

This is the seventh reading. What has held throughout: three operands under a
four-name type, nothing downstream reconciling them, and — now — no arm in the
adjust pass that ever sees the value. What changed: the fix is not in
`adjust_struct`, and §7.34's and §7.35's framings of it as an arm or walk
problem are both retired.
### 7.37 Root cause: the tag lookup misses a nested struct's qualified name

The inconsistent literal is in the symbol table after all — as the value of
`sol:@C@Base@book#11`:

```
Type........: struct Book
Value.......: { .title=0, .author=0, .book_id=0 }
```

Three initialisers, no pad, under a type whose padded layout has four members
(§7.32). So the pass *does* walk it, and `adjust_struct` *is* dispatched on it.

What it does next is resolve the padded layout by name:

```cpp
const symbolt *tag =
  context.find_symbol("tag-" + to_struct_type(t).name.as_string());
if (tag == nullptr || !tag->is_type)
  return;
```

The literal's struct type is named **`struct Book`**, and the only tag symbols
in the table are **`tag-struct Base.Book`** and `tag-Base` — the nested
struct's tag is qualified by its enclosing contract. So
`find_symbol("tag-struct Book")` misses, the arm returns before padding
anything, and the literal keeps three operands under a four-member type.

That is the root cause, and it explains §7.36's silence exactly: the probe sat
*after* the tag lookup, so a literal that bails at the lookup is never logged. The
arm's own comment says the padded layout "lives on the tag symbol; resolve by
name to reach it" — which is right, and the name it builds is wrong for a
nested struct.

It also explains the asymmetry in §7.32: the three operational-model structs
(`BytesStatic`, `BytesDynamic`, `BytesPool`) are top-level, so their tags are
`tag-struct BytesDynamic` and the lookup succeeds; only a struct declared
inside a contract gets the qualified tag.

The fix is at the lookup, not in the padding: resolve the tag for a nested
struct by the name its symbol actually carries. `ns.follow` already resolves
the type, so the qualified name is available — and a lookup miss should not
silently leave a value its own type cannot describe, which is the second half
of the repair and what
#7758's guard would then never need to fire for.

Seven readings before this one. Each was corrected by a measurement, and the
last of them was misled by where I placed the probe rather than by what the
code does — a probe after an early return cannot see the case that takes it.
### 7.38 507 of 507: the corpus agrees on every row (2026-09-13)

The fix is at the lookup §7.37 named, and it removes the lookup rather than
correcting the name:

```cpp
typet legacy = migrate_type_back(t);
add_padding(legacy, ns);
const type2tc padded = migrate_type(legacy);
```

`add_padding` is the same function that gave the tag symbol its layout, and
`clang_c_adjust::adjust_type` asserts it is idempotent, so a type already
carrying its pads is unchanged. No symbol, no name, nothing for a frontend's
tag-qualification convention to break. Correcting the name would have left
`adjust_struct` needing to know how each frontend qualifies nested tags — and
the struct type itself carries only the unqualified name, so that knowledge has
nowhere to come from.

| | §7.10 | §7.23 | §7.26 | now |
|---|---:|---:|---:|---:|
| verdicts agree | 441 | 441 | 504 | **507** |
| diverge | 67 | 63 | 0 | **0** |
| crash | 3 | 3 | 3 | **0** |

All three rows reach the *correct* verdict, not a masked one: `struct_1`
SUCCESSFUL, `struct_2` and `interface_7` FAILED, each matching the default
path.

`irep2_only_nested_struct_pad{,_fail}` pin it, and both halves flip to `tuple
field out of range` when the tag lookup is restored — so neither verdict is
satisfiable without the fix. The test declares its struct *inside* a contract
with a `string` member on purpose: a top-level struct resolves its tag under
either code path and would pin nothing. #7758's guard is what makes that
mutation legible; without it the mutant would SIGSEGV and the cause would have
to be inferred again.

Nets: `irep2_only` 111 of 111, unit 871 of 871, and the Solidity default path
unchanged at 523 of 525 with the two `KNOWNBUG` rows that already passed.

**What this section cost, and what it bought.** Eight readings of one defect, seven of
them wrong in some particular: an ill-sorted assignment, a missing member name,
a stale sort, a nested type symbol, an inline snapshot on an expression, the
arm's bail-out, the dispatcher's walk. Each was retired by an instrument built
so that silence or a surprise would contradict it — and one reading was wrong
only because a probe sat *after* the early return it needed to observe (§7.36).
The answer was finally in a symbol-table dump and a `grep` of tag names, not in
a ninth probe. The by-product was #7758: `project` read past the end of a
vector in every release build, guarded only under `!NDEBUG`, which is a class
of undefined behaviour with reach far beyond this corpus.
### 7.39 The shared arm verified on all three consumers, and S.3 priced

`adjust_struct` is a **C** arm, so §7.38's change runs for the C and C++
frontends under their flags too, and it is behaviour-altering exactly where the
old tag lookup used to fail: such literals now get padded where they were
previously left alone. The default path is unaffected by construction — the arm
table is entered only under `sole_adjuster` — so the exposure is the two flag
paths.

C was covered by `irep2_only` (111 of 111). C++ was not, and that corpus is not
at verdict parity, so a regression there would surface as a new hard failure
rather than a verdict change. Stride-12 sample of `esbmc-cpp/cpp` under
`--clang-cpp-irep2-adjust-only`, counting SIGSEGV, `tuple field out of range`
and uncaught exceptions:

```
--- 84 sampled: 0 hard failures ---
```

So the shared arm is verified on all three consumers.

**S.3, priced.** The converter's header declares:

| shape | count |
|---|---:|
| functions taking `exprt &` | 136 |
| of those, named `new_expr` | 67 |
| functions taking `typet &` | 56 |

So the converter is not a set of value-returning builders that could be ported
one at a time: 136 entry points write through an `exprt &` out-parameter, 67 of
them the same `new_expr` that `get_expr` threads through the whole expression
walk. Porting any one of them natively leaves its caller holding an `expr2tc`
where an `exprt` is expected, which is why §7.25's boundary question has no
local answer.

That reframes S.3's decomposition. It is not "port the smallest file first" —
the smallest file's five functions all write into the same `new_expr` as the
largest — but "convert the out-parameter", and the unit of work is the
out-parameter's transitive closure rather than a file. The three options in
§7.27 are the choices for how that closure's *edge* behaves, and the cheapest
edge is the one that does not exist: port `new_expr` itself, which means the
converter moves in one change or not at all.

That is a materially different plan from §3's step S.3, and the parent's §6
ordering put Solidity before Python precisely so a phase like this could be
sized honestly before it starts. Recording the measurement rather than a
porting order, because the measurement says the order does not matter.
## 4. The corpus swept under asserts, and two ways to measure it wrong

The flag this frontend honours is Phase 7's, so this is the first divergence count
for it. Run under `DebugOpt`, so with asserts, over every test in
`regression/esbmc-solidity`.

**Result: 513 agree, 1 diverges, 1 aborts, 10 have no source to run.**

Both non-agreeing rows are characterised below, but the number took three attempts
to produce, and the two wrong ones are worth recording because each looked
authoritative.

### 4.1 First wrong answer: 514 divergences

The Solidity frontend extracts to `/tmp` with a random suffix per run, so the
source paths in a GOTO dump differ between *any* two runs -- flag or no flag. The
first sweep therefore reported almost every row as diverging. Normalising
`/tmp/esbmc_solidity_temp-[0-9a-f-]+` to a fixed token is required before any
A/B of this frontend, and the same applies to python.

### 4.2 Second wrong answer: 9 divergences

Eight of those nine were `irep2_only_*` tests, whose descriptors *already* pass
`--clang-cpp-irep2-adjust-only`. The sweep appended it a second time, and ESBMC
does not tolerate that: the run collapses from 7 057 lines of output to one. So
the A/B is meaningless for any test that already enables the flag -- there is
nothing to compare, and passing it twice does not produce the same run.

Skip those rows rather than comparing them.

### 4.3 The one genuine divergence

`nested_array_mixed_1`, eight lines, one shape -- the target type of a cast
applied to `__ESBMC_array_push`'s result:

```
- ASSIGN this->mixed[0]=(unsigned _ExtInt(256) [4] *)return_value$...
+ ASSIGN this->mixed[0]=(unsigned _ExtInt(256) * *)return_value$...
```

Legacy casts to pointer-to-array-of-4; the IREP2 pass casts to
pointer-to-pointer. The two are not interchangeable -- the pointee size differs,
which is what pointer arithmetic and the next dereference read -- so this looked
like the one row worth closing.

**It is the legacy side that is wrong.** Three things in the same symbol-table dump
settle it, all identical on both paths:

```
this->mixed = (unsigned _ExtInt(256) * * *)(calloc(2, sizeof(... * *)));
... sizeof(unsigned _ExtInt(256) [4]) ...
```

`this->mixed` is `T***` on both paths, so `this->mixed[0]` is `T**`. The IREP2
pass casts the pushed result to `T**`, which is the type of the lvalue it is
assigned to; the legacy pass casts it to `T[4]*`, which is not. The `sizeof` the
call is given is the same either way, so the allocation is unaffected.

That is the same shape as the python row in `frontends-to-irep2.md` §40.4, and the
same conclusion: the flag-on side agrees with the type system and the default path
is the outlier. So Phase 8's corpus has **no row where the IREP2 pass is wrong** --
513 agree, and the one that differs differs in the IREP2 pass's favour.

### 4.3a Where the fix is not

The obvious one-line fix is wrong, and measuring it costs less than arguing about
it. The converter builds the assignment from the Solidity-declared type:

```cpp
exprt tmp = side_effect_exprt("assign", base_t);
convert_type_expr(ns, new_expr, base_t, expr);
```

Substituting `base.type()` -- convert to the lvalue rather than to the declaration
-- changes nothing, because `base.type()` *is* `T[4]*` at that point. The converter
emits the declared shape for both the member and the cast.

What separates the paths is later: **both** adjust passes lower the member to
`T***`, and only the IREP2 one also lowers the cast the converter left at `T[4]*`.
So the legacy pass is internally inconsistent -- it lowers the lvalue's type and
not the cast feeding it -- and closing the row means changing how the legacy adjust
lowers an array-typed pointee, which reaches every such cast on the default path.
That is wider than a converter tweak and wants its own measurement.

Changing the default path is a behaviour change for every nested-array push, so it
stays its own PR. What is settled is which side would be changed, and now also
that the change is not where it first appeared to be.

### 4.4 The abort is not ours

`delegate_shadow_3` trips `member2t`'s component assertion at
`irep2_expr.h:1641`. It does so **identically on both paths**, and its descriptor's
first line is already `KNOWNBUG`. Recorded so the next sweep does not read it as a
hop-off failure.

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

## 17. The blocker was the printer, not the seam (2026-09-16)

§16.3 left the ten deferred writes behind one question -- whether `migrate_expr_back` should
reconstruct a level0 symbol's `name` -- and noted a narrower candidate. The narrower one is the right
one, and it is a defect in its own right rather than a migration concession.

### 17.1 The wrong question, asked for years

`c_expr2stringt::get_shorthands` decides whether a shorthand is ambiguous by comparing whole
expressions:

```cpp
// c_expr2string.cpp:52 (before)
if (result.first->second != symbol)
{
  ns_collision.insert(symbol.identifier());
  ns_collision.insert(result.first->second.identifier());
}
```

It is worse than an imprecise test: it is a **tautology**. `symbols` is a `std::set<exprt>`, ordered
by `compare()`, and `compare()` and `operator==` ignore exactly the same thing -- comments
(`irep.cpp:186-205`, "comments are NOT checked", and `:322-375`). So the set deduplicates precisely
the pairs for which `operator==` holds, every pair of *distinct* elements is unequal, and the guard
was true wherever it was evaluated. The pre-fix code marked a collision on every shorthand clash and
decided nothing.

That also settles the question the change invites -- whether it can lose real disambiguation. There
was no decision being made to lose. And in the one case the fix alters, the old output carried no
information either: when two spellings share an identifier, `ns_collision` holds that single
identifier, so `convert_symbol` prints the mangled form for *both* occurrences. `x … x` became
`c:@F@f@x … c:@F@f@x` -- the same string twice, only longer.

Comparing identifiers is the first version of this code that can express the intended distinction.
Differing identifiers mean two distinct symbols competing for one shorthand, a real collision;
equal ones mean **one** symbol wearing two spellings, which is not.

Comparing identifiers instead is sound because the identifier is the unique key for storage:
`symbol2t::get_symbol_name` is "a pure function of the symbol's (thename, rlevel, l1, thread, node,
l2) identity fields" (`irep2_expr.cpp:130-145`), so SSA renaming is *inside* the name and two
instances never share one. `get_symbols` collects only `id() == "symbol"`
(`c_expr2string.cpp:31-38`), so `next_symbol` and `nondet_symbol` -- which could share an identifier
while denoting different values -- never enter the map.

### 17.2 It is live, not latent

The old code flagged all 5864 clashes. Instrumenting it with the identifier comparison the fix
introduces splits them: **5852 where the identifiers differ and 12 where they do not**, the latter in
twelve named tests
(`address_bind_3/4/5/7`, `array_1/2`, `import_10`, `modifier_8`, `reentrance_12`, `tuple_6`,
`unbound_5/7`). So this was never only a migration blocker. In `array_1`:

```
before:   sol:@x#4=1;   y=sol:@x#4;
after:    x=1;          y=x;
```

### 17.3 Why it needed more care than a one-line diff suggests

Two invariants sit on this function, and neither is obvious from the call site.

`from_expr` is a documented interface surface: `goto_coverage.cpp:818-828` states that any change
altering its formatting "must preserve this 1:1 mapping or the percentage will silently deflate",
because a k-path claim's idf string is built from printed text and `ns_collision` is per-printer-
instance. Printing more short names is exactly the direction that could collapse two claims. And
`witnesses.cpp:949-980` builds witness assignments through `from_expr`, which SV-COMP validates --
hence `needs-svcomp-run` on the change. `parse_result` in `esbmc-wrapper.py` matches verdict lines and
violated-property text only, never variable names, so classification cannot move.

That suite is genuinely sensitive rather than incidentally green: 67 of its 144 descriptors pin a
coverage percentage, 19 exercise k-path and 8 pin a `Spanning Set` line. All 144 pass.

The third consumer matters most and is the least obvious. `goto2c::expr2ct` inherits this same
`get_shorthands` and overrides `convert_symbol` to sanitise a mangled id into `c__F__f__x`
(`goto2c/expr2c.cpp:484-505`), and goto2c emits C that has to compile -- with declarations printed per
symbol in separate calls, so a declaration always took the short name while a use containing a double
spelling took the mangled one, naming an identifier the generated program never declared. The fix
removes that mismatch: `goto-transcoder` 268/268.

Measured: `goto-transcoder` 268/268, `goto-coverage` 144/144, `witnesses` 163/163, unit 883/883,
`esbmc-solidity` 526/526, `esbmc-cpp/cpp` 1065 with only its six known pre-existing failures.

One thing the witness figure does *not* establish: those descriptors pin verdicts, and the 124 under
`witnesses_validate/` consume a witness rather than produce one, so 163/163 says no verdict moved --
not that produced witness text is unchanged. What argues the direction is safe is that
`get_formated_assignment` (`witnesses.cpp:939-959`) calls `from_expr` once for the lhs and once for a
`is_constant_expr`-guarded value, and a single-symbol call cannot clash; and that when the old code did
fire it emitted `c:@F@main@x`, which is not a valid C identifier for a validator to parse at all.

The whole C suite exceeds the ten-minute cap, so rather than sample it the at-risk set was selected by
what a change to name printing can actually break -- a `test.desc` whose *expected* output embeds a
mangled or qualified name:

```sh
awk 'FNR>3 && /@|::/ {print FILENAME}' $(find regression -name test.desc -not -path "*/disabled/*")
```

That is 188 tests, 129 of them `ir-ra`, spread over ten suites. All 188 pass apart from `ch8_5` and
`github_7433_library_fail`, both in the six pre-existing failures above. A suite-level cap forces a
choice of subset; choosing it by the property under test beats choosing it by index.

### 17.4 The pin is two directions, not two verdicts

No verdict moves, so a `SUCCESSFUL`/`FAILED` pair would pin nothing. What can go wrong here is
one-sided in each direction, so `regression/esbmc-solidity/shorthand_spurious` asserts both in a
single dump: `^x=1;$` and `^y=x;$` (the spurious collision must not fire) alongside
`^sol:@Base=&"Base"\[0\];$` (a genuine one must still fire). Mutation-checked twice -- reverting the
condition fails it, and replacing it with `if (false)`, which suppresses every collision, also fails
it. The test cannot pass with the fix absent or over-applied.

With this in, the ten writes §16.2 deferred lose their only objection, and re-attempting them is the
next step.
