/* Supporting-invariant synthesis lets the ranking checker certify loops
 * whose decrease obligation needs a relational fact about other body-
 * modified variables. The guard-derived measure is m = x; the body
 * decreases x by d1 each iteration, so decrease requires d1 > 0 — a fact
 * the bare ranking checker cannot establish since d1 becomes nondet after
 * havoc.
 *
 * The invariant synthesizer seeds candidate bounds from constant pre-loop
 * assignments (d1 = 73, d2 = 74) in the function's straight-line prefix —
 * which is its only seeding path, so dominance is conservative — then
 * keeps only atoms inductive under the body. Both lower bounds survive
 * (d1' = d2 + 1 >= 74 under d2 >= 74; d2' = d1old + 1 = d1 + 1 >= 74 under
 * d1 >= 73), so I = (d1 >= 73 /\ d2 >= 74) is assumed and the decrease
 * obligation `x > 0 /\ x - d1 >= x` becomes UNSAT — the checker certifies.
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int d1 = 73;
  int d2 = 74;
  int d1old;
  while (x >= 0)
  {
    x = x - d1;
    d1old = d1;
    d1 = d2 + 1;
    d2 = d1old + 1;
  }
  return 0;
}
