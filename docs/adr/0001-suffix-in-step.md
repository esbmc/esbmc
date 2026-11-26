# The suffix is part of the step

Code after the single loop (the suffix) runs inside the step that takes the
exit: its assertions become bad properties guarded by the exit condition, and
an exit step has no successor. We did not add a "done" location bit, which
would make bad a pure state predicate, because it costs a state bit, a longer
counterexample, and changes to every consumer, and it only pays off once
there is more than one loop. Multi-loop support kept this shape: see
[0002](./0002-multi-loop-by-flattening.md), which flattens several loops into
one so the suffix stays part of the step.

## Consequences

- Bad properties may now also read variables that only the suffix defines.
  They already read step inputs, so bad was never a pure state predicate.
- The loop-entry condition that k-induction assumes after the havoc is no
  longer an invariant: it is false in exit steps.
- Every call in the loop function is checked, suffix included, and a callee
  that k-induction transformed is refused: in TS mode its loop has no entry
  flag, so the assume after its havoc is false when the loop is not entered.
