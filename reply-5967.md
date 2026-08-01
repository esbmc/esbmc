Short version up front: the three questions come from reading "base case" and "inductive step" as one thing, but in this run they are two different things, and separating them answers all three. Everything below is measured on your exact program with ESBMC 8.4.0 — commands are at the end so you can reproduce it.

## Two independent meanings of "base case / inductive step"

1. The two invariant obligations that `__ESBMC_loop_invariant(res==idx)` gets compiled into. `--show-claims` lists them as two separate properties:
   - **Claim 1 — loop invariant base case** → does the invariant hold on entry?
   - **Claim 2 — loop invariant inductive step** → does one body iteration preserve it?
2. The three phases of the k-induction algorithm, run at every unwind bound k: base case → forward condition → inductive step (`do_bmc_strategy`, `src/esbmc/parseoptions/bmc_strategy.cpp`).

Both invariant obligations are emitted together, side by side, inside a single straight-line verification branch that the combined pass (`goto_loop_invariant_combined`) prepends to the loop. Here is your instrumented `main` (`--goto-functions-only`, trimmed):

```
        ASSIGN idx=0; ASSIGN res=0;
   L0:  IF !(NONDET(_Bool)) THEN GOTO 1        // nondet gate around the check
        ASSERT res == idx        // Claim 1: loop invariant base case
        ASSIGN idx=NONDET; ASSIGN res=NONDET   // havoc
        ASSUME res == idx                      // assume invariant
        ASSUME idx < 10                        // assume loop entry cond
        ASSIGN res=res+1; ASSIGN idx=idx+1     // ONE body iteration
        ASSERT res == idx        // Claim 2: loop invariant inductive step
        ASSUME 0                               // dead-end: never falls through
    1:  ASSIGN idx=NONDET; ASSIGN res=NONDET   // k-induction havoc  (inductive_step only)
        ASSUME idx < 10                        // entry cond         (inductive_step only)
    2:  IF !(idx < 10) THEN GOTO 3             // the REAL loop
        ASSUME res == idx                      // invariant re-asserted as a hint
        ASSIGN res=res+1; ASSIGN idx=idx+1
        GOTO 2
    3:  ASSERT res == 10                       // your post-loop assert
        ASSERT res == idx                      // your post-loop assert
```

**Key point:** that verification branch (L0…ASSUME 0) is k-independent — it havocs, does exactly one iteration, and dead-ends on `ASSUME 0`. It is re-encoded identically at every k. The unwind bound only affects the real loop at L2 and the two post-loop asserts.

## What each phase actually checks at k = 6 (measured)

Running the real pipeline and reading off the surviving VCCs per phase (`--show-vcc`, counts from "Generated N VCC(s), M remaining"):

| k-induction phase | live proof obligations at k=6 |
|---|---|
| base case | loop invariant inductive step |
| forward condition | loop invariant inductive step, unwinding assertion loop 3 |
| inductive step | loop invariant inductive step, assert res==10, assert res==idx |

Two facts that explain the whole table:

- **loop invariant base case (Claim 1) never becomes a VCC** — 0 occurrences in any phase, any k. At that program point `res` and `idx` are both literally 0, so `res==idx` folds to `0==0` and is discharged by constant propagation before it ever reaches the solver. It is still Claim 1; it just carries no obligation in this program.
- The base case and forward condition **skip the L1 havoc** (`execution_state.cpp:185`: inductive_step_instructions are not executed unless we are in the inductive step). So in the base case the real loop runs from the concrete `idx=0`; at k<10 it is truncated by `ASSUME(idx>=10)` and the post-loop asserts are unreachable — hence they are not VCCs there. Only the inductive step havocs `idx`/`res`, so only there do the post-loop asserts (and the invariant hint) become live obligations. The forward condition additionally runs with `--no-assertions`, which drops **user-provided** asserts (`res==10`, `res==idx`) but not the compiler-inserted invariant self-check (`symex_main.cpp:482`).

## Your three questions

**Q1 — obligation for the invariant inductive step when checking the base case at k=6.** It is a live obligation (in fact the only one in the base-case phase): "from a state with `res==idx ∧ idx<10`, after one iteration `res++; idx++`, does `res==idx` still hold?" — which is valid (`res+1==idx+1`). It is checked at every k, not specially at 6, because that self-check branch is k-independent.

**Q2 — obligation for the invariant base case when checking the induction case at k=6.** None, in this program — it is present in the inductive-step encoding too, but `res==idx` at loop entry is `0==0` and is folded away, so it generates no VCC. (If the pre-loop values were non-constant it would be a real obligation: "the invariant holds on entry.") The inductive-step phase's actual live obligations at k=6 are the invariant inductive step plus your two post-loop asserts `res==10` and `res==idx`.

**Q3 — does ESBMC skip the invariant base case for k>6 once the invariant inductive step passes at k=6?** No. There is no per-property skipping across k: every k re-encodes and re-checks the full claim set — the k=7 base-case encoding still contains the same verification branch with both invariant asserts (identical VCC set to k=6). An individual assert passing never terminates or prunes k-induction; the algorithm stops only when a phase is conclusive (base case SAT → FAILED; forward condition UNSAT, or inductive step UNSAT → SUCCESSFUL). So for k>6 the base case keeps running and keeps re-checking the same obligations.

## The thing that's probably really going on

With the default settings this program is proven SUCCESSFUL at k=11, by the forward condition — i.e. once the bound reaches the loop's real trip count (10) the loop is fully unwound and all states are shown reachable. The `res==idx` invariant is inductive (its self-check passes at every k) but it is too weak to let the inductive-step phase discharge `res==10`, so the inductive step stays SAT and never closes the proof early. Strengthening it to `res==idx && idx<=10` does not change this — it still finishes at k=11 via the forward condition, and so does the same program with no invariant at all. In other words, for this example the invariant is inert: BMC + forward condition is what proves it. If you cap `--max-k-step` below 11 (e.g. 6 or 8) you get VERIFICATION UNKNOWN, which is likely the intermediate behaviour that prompted the question.

## Reproduce

```sh
esbmc inv.c --loop-invariant --k-induction --show-claims          # the 4 claims
esbmc inv.c --loop-invariant --k-induction --goto-functions-only  # the branch above
# per-phase VCCs at a fixed k (streams separated so phases don't interleave):
esbmc inv.c --loop-invariant --k-induction --base-k-step 6 --max-k-step 7 --show-vcc \
      1>vcc.stdout 2>vcc.stderr
grep -E 'Checking|Generated' vcc.stderr          # phase + count per encoding
awk '/^file /{getline c; print c}' vcc.stdout     # identity of each surviving VCC
esbmc inv.c --loop-invariant --k-induction                        # default: SUCCESSFUL at k=11
```
