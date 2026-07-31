# Scope — V.2 / W3 IREP2-native attribute carriage

> **Status: re-scoped to Option D; all three steps landed (2026-07-31, §9-§10).**
> `docs/roadmap/irep2-migration.md` §V.7 lists V.2/W3 as one of two remaining
> items — *"Untouched; the highest shared blast radius left in the program"* —
> and V.1a and V.6 both depend on it. This scope re-grounds §V.2 against the
> tree (2026-07-30) and finds that **the design §V.2 prescribes is refuted by an
> argument Part III already recorded and never propagated forward.**
>
> §1-§8 are the original forward plan and its re-scoping argument, unchanged.
> §9-§10 record what actually landed.

## 1. What §V.2 asks for

Verbatim (`irep2-migration.md:4421-4428`):

> Move `#cpp_type`/`#member_name`/`#cformat` off `irept` onto a **typed IREP2
> companion keyed by symbol id** (the §5.1 pattern — *not* a generic string-map,
> which re-introduces the escape hatch IREP2 abolished). Teach the three
> external consumers (`clang_cpp_adjust_expr`, `cpp_expr2string`,
> `goto2c/expr2c`) to read the IREP2 form.

Two of that paragraph's load-bearing claims do not survive contact with the
tree: the consumer count, and the keying.

## 2. The consumer census (re-derived)

`git grep -c '#cpp_type' -- src` and siblings. **Reads**, classified by hand —
the raw counts include writes, accessor bodies and comments:

| reader | attribute | what it does |
|---|---|---|
| `clang-cpp-frontend/clang_cpp_adjust_expr.cpp:582` | `#cpp_type` | builds exception type ids for catch matching |
| `util/lang/cpp_expr2string.cpp:138,140` | `#cpp_type` | counterexample type spelling (`signed char`, …) |
| `goto2c/expr2c.cpp:174` | `#cpp_type` | distinguishes same-width types (`long` vs `long long`) when emitting C |
| `clang-cpp-frontend/clang_cpp_adjust_code_gen.cpp:51,200` | `#member_name` | ctor member lookup + symbol-name construction |

**Correction to §V.2: there are four reader sites, not three.**
`clang_cpp_adjust_code_gen.cpp` is a real consumer and is not in §V.2's list.
It is also the *only* `#member_name` reader, which matters because it makes
`#member_name` separable from `#cpp_type` (§5).

Two further corrections:

- **`#cformat` has no external consumer at all.** Its only non-frontend
  presence is the `irept::cformat()` accessor itself (`util/irep/irep.cpp`,
  `util/irep/std_expr.h`), which Phase 4.1 already routed Python through. Its
  remaining writers are Solidity (`solidity_convert_type.cpp:2`,
  `solidity_convert_literals.cpp:1`). Bundling it with the other two overstates
  the work.
- `goto-programs/exception_typeid.h` matches a `#cpp_type` grep but the hit is
  a **comment**, not a read. It is not a consumer.

### 2.1 The writer side is four frontends, not one

§V.2 sits in Part V (Python), but the attributes are written by four frontends
at very different levels of encapsulation:

| frontend | encapsulation state |
|---|---|
| Python | **encapsulated** — Phase 4.1 routed all access through `type_utils::{set,get}_cpp_type` / `{set,remove}_member_name` (`type_utils.h`); raw access is the accessor bodies only |
| Solidity | **partial** — Phase 3.1 encapsulated the `#sol_*` family, but `#cpp_type` (`solidity_convert_type.cpp:7`, `solidity_convert.cpp:1`, `solidity_convert_literals.cpp:1`) and `#member_name` (7 sites across 5 files) were never in scope |
| clang-cpp | **raw** — `clang_cpp_convert.cpp:3`, `clang_cpp_convert.h:1` write `#member_name` directly |
| clang-c | **raw** — `clang_c_convert.cpp:1891` writes `#cpp_type` |

So a Python-scoped V.2 cannot remove the attributes: three other frontends keep
writing them, and the four readers keep reading them. **V.2 is only meaningful
as a repo-wide change**, which is why §V.7 correctly calls it the highest
shared blast radius left.

## 3. The decisive finding — §V.2's keying is already refuted

§V.2 prescribes "a typed IREP2 companion **keyed by symbol id**". Part III
§14's Q-S1 investigated exactly that design for Solidity's `#sol_*` and killed
it:

> the reads are **value-carried, not symbol-reachable** … ~40 of the ~43 read
> sites read off a transient `typet` value or an expression's `.type()` — with
> **no associated symbol at read time**.

**Every W3 reader has the same shape.** Verified by reading each site:

- `cpp_expr2string.cpp:138` — `src` is the type currently being printed,
  reached *recursively* (`convert(src.subtype())`). There is no symbol; worse,
  the metadata must survive subtype traversal.
- `goto2c/expr2c.cpp:174` — `src` is a type being named; the read is guarded on
  `width % 8 == 0`, i.e. it is a property of the type value.
- `clang_cpp_adjust_expr.cpp:582` — `type` is a catch parameter type in the
  middle of id construction.
- `clang_cpp_adjust_code_gen.cpp:51` — `ctor_type.get("#member_name")` is read
  off a value and then *used* as a lookup key; the symbol is the result, not
  the context.

A symbol-keyed side table cannot serve any of them. Part III's conclusion
transfers verbatim: **the metadata must travel with the value**, so only a
value-bundled companion can carry it — and that companion "re-introduces the
attribute flexibility IREP2 removed, for no verifier-core benefit".

This is not a new discovery. It is Part III's finding, which was recorded in
Part III and never propagated to §V.2 — the plan text predates it. §V.2 as
written should be treated as **superseded**.

## 4. Why this also re-validates B1

Part III §14 closed with the Solidity frontend staying legacy "by design",
citing B1. The same argument applied to W3 lands in the same place, and for a
sharper reason: `#cpp_type` exists precisely to carry information IREP2's
closed type system **deliberately discards** — the source-level *spelling* of a
type. `long` and `long long` are the same `signedbv_type2t` at the same width;
that is not a gap in IREP2, it is the normalization IREP2 was built to perform.
`goto2c/expr2c.cpp:160-173` says so in its own comment.

The three `#cpp_type` readers are, without exception, **presentation** consumers
— counterexample text, generated C, and exception-id strings. None is verifier
core. Removing W3 buys no soundness and no round-trip.

## 5. Options, and the recommendation

| # | Option | Cost | Verdict |
|---|---|---|---|
| A | Symbol-keyed IREP2 companion (as §V.2 specifies) | — | **Not viable.** §3 |
| B | Value-bundled companion wrapping `type2tc` | high | Viable but re-creates attribute flexibility; rejected on the same cost/benefit as Part III §14 |
| C | Extend `type2t` with a spelling field | medium | Explicitly rejected upstream ("extending `type2t` (rejected)", Part III §14) — and it would push presentation concerns into the verifier IR |
| D | **Encapsulate the remaining raw writers; leave carriage legacy** | low (2-3 PRs) | **Recommended.** §5.1 |
| E | Do nothing | zero | Defensible, but leaves four frontends writing raw ireps with no repoint-point |

### 5.1 Option D — what it actually is

Both Part III and Part IV independently converged on the same end state:
*encapsulate the attribute behind one typed seam per frontend, leave the
storage legacy.* Python is already there (Phase 4.1); Solidity is there for
`#sol_*` (Phase 3.1). Option D finishes the pattern for the attributes and
frontends that were never in scope:

1. **clang-cpp `#member_name`** (`clang_cpp_convert.cpp`, `.h`) — 4 sites
   behind a typed accessor. Pairs with the single reader at
   `clang_cpp_adjust_code_gen.cpp:51,200`, so `#member_name` becomes fully
   seamed end-to-end in one PR. **DONE — §9.**
2. **Solidity `#cpp_type` + `#member_name`** — the sites Phase 3.1 skipped,
   plus the two `#sol_*` stragglers `irep2-migration.md` §14's correction names
   (`#sol_tuple_id`, `#sol_unchecked`). **DONE — §10** (the `#sol_*` pair
   excepted, see §10.2).
3. **clang-c `#cpp_type`** (`clang_c_convert.cpp:1891`) — one site.
   **DONE — §10.**

This delivers the only property V.2 was actually going to buy — *one
repoint-point per attribute per frontend* — at a small fraction of the cost,
and it leaves B/C available should the cost/benefit ever change. It is a
behaviour-preserving mechanical no-op per PR, gated exactly as Phase 3.1 was.

**What Option D does not do:** it does not remove W3, so it does **not**
unblock §V.1 bar #4, and it does not unblock V.1a or V.6. Those stay blocked,
now with a recorded reason rather than an unexamined plan. That is the honest
trade and it should be stated in any PR that takes this option.

## 6. Gates (for Option D)

| # | Gate |
|---|---|
| G1 | Per PR: same key, same value, same storage — a mechanical no-op |
| G2 | Verdict parity on **`esbmc-cpp`, `esbmc-solidity` and `python`** — these consumers serve C++ and Solidity too, so a Python-only gate is insufficient |
| G3 | Counterexample **text** parity, not just verdicts — `cpp_expr2string` is a reader, so its output is asserted verbatim by `test.desc` regexes |
| G4 | `goto2c` output parity on the `goto2c` suite |
| G5 | Raw-access census after each PR: `git grep -n '#cpp_type\|#member_name\|#cformat' -- src/<frontend>` returns accessor bodies only |

**Environment caveat.** `esbmc-solidity` needs `solc` and cannot run on macOS
(`sol64` operational models are stubbed empty — `_BitInt(256)` is unavailable on
Apple aarch64). Gate G2 must run on Linux CI. This bit Phase 3.1 and is
recorded in `irep2-migration.md` §14.

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Option D is mistaken for progress toward removing W3 | Say so explicitly in each PR body; §V.7 bar #4 stays "no" |
| R2 | An accessor changes the value's spelling or default | G1: byte-identical key and value; mechanical no-op only |
| R3 | A reader is missed and keeps raw-reading | G5 census is repo-wide, not frontend-scoped |
| R4 | Someone revives Option A from the stale §V.2 text | §3 of this document; `irep2-migration.md` §V.2 should link here |

## 8. One-line summary

§V.2's symbol-keyed companion cannot work — all four W3 readers read off
transient type *values*, the same Q-S1 wall Part III hit — so the honest choice
is the low-cost encapsulation pass that Parts III and IV both converged on,
with W3 removal declined on a recorded rationale rather than left implicitly
open.

## 9. Option D step 1 — what landed (2026-07-31)

`#member_name` is now seamed end-to-end: both clang-cpp writers
(`clang_cpp_convert.cpp:2762,2831`), both readers
(`clang_cpp_adjust_code_gen.cpp:51,200`), and the Python frontend's existing
`type_utils` accessors all go through one place.

**One shared seam, not one per frontend — a deliberate departure from §5.1.**
The accessor is `irept::member_name()` / `member_name(val)` /
`remove_member_name()`, keyed by an interned `a_member_name`, alongside the
`irept::cformat()` pair that already treats this attribute family exactly this
way (`irep.h:478`). Two reasons the shared form is strictly better here:

- **The reader is shared.** `clang_cpp_adjust_code_gen` reads what clang-cpp,
  Solidity *and* Python write. A per-frontend seam would still leave the key
  literal in three writer headers plus the reader — four repoint-points for one
  attribute, which is not what §5.1 was trying to buy.
- **It matches the codebase's own convention**, so step 2 (Solidity) becomes
  seven one-line call-site edits with no new accessor to design.

After this step the only live `#member_name` key literal in the tree is
`irep.cpp:508`. The remaining `git grep` hits in `src/clang-cpp-frontend` and
`src/python-frontend` are comments.

### 9.1 Gates

| # | Result |
|---|---|
| G1 | **Discharged.** `--goto-functions-only` A/B against master over all 582 runnable `esbmc-cpp/cpp` tests: 2 raw divergences, both non-attributable — `ch19_1` embeds a wall-clock timestamp in a string literal, `cpp_sum_class` differs only in stdout/stderr interleaving of a progress line |
| G2 | **Discharged** for `esbmc-cpp` (676/685; the 9 failures reproduce identically on the master binary) and `python` (375/375 on a stride-4 slice). `esbmc-solidity` is untouched by this step and is macOS-blocked — it rides CI |
| G3 | Not applicable to this step: `cpp_expr2string` reads `#cpp_type`, not `#member_name` |
| G4 | Not applicable: `goto2c/expr2c` reads `#cpp_type` |
| G5 | **Discharged** for clang-cpp and Python (comments only). Solidity's 7 writers are step 2 |

**Harness note for step 2.** Both A/B censuses on this track were first invalidated
by randomized temp directories embedded in the GOTO dump, in four spellings —
`esbmc.XXXX-`, `esbmc_XXXX-`, `esbmc-cpp-headers-`, `esbmc-headers-` — plus
`esbmc-python-astgen-` on the Python side. Normalize the whole path segment
(`s@/esbmc[-._][^/ ]*@/TMPD@g`), not individual prefixes, or the run reports
~90 % false divergence.

### 9.2 What this does not do

Unchanged from §5.1's honest trade, restated because it is easy to lose: this
does **not** remove W3. §V.1 bar #4 stays "no", and V.1a and V.6 stay blocked.
What it buys is one repoint-point for `#member_name` if carriage is ever
revisited — and §3's argument that it should not be is unaffected.

## 10. Option D steps 2 and 3 — what landed (2026-07-31)

Steps 2 and 3 shipped together, because splitting them would have introduced a
`cpp_type()` accessor used by one frontend while another kept writing the key
raw. `irept::cpp_type()` / `cpp_type(val)` joins `member_name()` on the same
interned-key pattern, and every remaining site now goes through one of the two:

| what | sites |
|---|---|
| Solidity `#member_name` writers | 8 across 6 files (`solidity_convert_{builtin,constructor,contract,decl,modifier,tuple}.cpp`) |
| Solidity `#cpp_type` writers | 7 (`solidity_convert.cpp:118`, `solidity_convert_type.cpp` ×6) |
| clang-c `#cpp_type` writer | 1 (`clang_c_convert.cpp:1891`) — this is step 3 |
| `#cpp_type` readers | 3 (`clang_cpp_adjust_expr.cpp:582`, `goto2c/expr2c.cpp:174`, `cpp_expr2string.cpp:138,140`) |
| Python `type_utils` cpp_type accessors | delegate rather than spelling the key |

**Option D is now complete.** `git grep '"#member_name"\|"#cpp_type"' -- src`
returns comments plus the two interned definitions in `irep.cpp:508-509`, and
nothing else. §5.1's three-step list is discharged.

### 10.1 Gates

| # | Result |
|---|---|
| G1 | **Discharged.** Full-output A/B against master over all 582 runnable `esbmc-cpp/cpp` tests: 1 raw divergence, `cpp_sum_class`, proven non-attributable by a **master-vs-master self-A/B control** that produces 2 140 diff lines on the same test — it is `--k-induction-parallel`, the shape §5's census rule 3 already flags as unstable against itself |
| G2 | **Discharged.** `esbmc-cpp` 676/685 — the same 9 failures as §9.1, each verified identical on the master binary; `python` green on a stride-5 slice. `esbmc-solidity` rides Linux CI (macOS has no `solc` and stubbed `sol64` models) |
| G3 | **Discharged, and it is the load-bearing one here** — `cpp_expr2string` is a reader, so the A/B compares full stdout including counterexample text, not just verdicts. 0 divergences over all 106 `esbmc-cpp/cpp/*_fail` tests, which are the ones that print a counterexample |
| G4 | **Not runnable.** `goto2c/expr2c.cpp` is a reader, but the tree has no goto2c regression suite and no `test.desc` invokes that path — `grep -rl goto2c regression/` is empty. The reader is covered by inspection only (a one-line `get` → accessor substitution). Recorded rather than claimed |
| G5 | **Discharged repo-wide**, see above |

**Harness note, extending §9.1's.** A *full-output* A/B additionally needs the
timing lines stripped — `Symex completed in: 0.000s` vs `0.001s` alone produced
46 false divergences out of 106. Filter `completed in:|time:|Runtime|Elapsed`
on top of the temp-path normalization, and exclude `--k-induction-parallel`
tests as §5 rule 3 requires. When a divergence survives, run the baseline
against *itself* before attributing it to the patch: that control is what
settled `cpp_sum_class` here, and it costs one extra run.

### 10.2 Residue

The two `#sol_*` stragglers §5.1 step 2 mentions (`#sol_tuple_id`,
`#sol_unchecked`) are a different attribute family from W3's and were left
alone; they belong with Phase 3.1's `#sol_*` work, not here.
