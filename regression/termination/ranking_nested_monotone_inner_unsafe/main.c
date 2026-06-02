// Soundness regression: the inner has a non-unit-step assign (r := r
// + 2). The monotone-counter refinement MUST NOT fire — its formula
// (post r == pre_r + N * 1) would over-count by half. With the
// detector restricted to ±1 steps, this falls back to pure havoc,
// and the OUTER's ranking will fail (no measure decreases under a
// fully havoc'd post-state).
//
// Pins that the refinement's step-size guard is honoured. Expected
// outcome: UNKNOWN (NOT SUCCESSFUL). The outer is in fact terminating
// (because r grows without bound and y stays bounded), but proving it
// requires reasoning we don't yet do.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  if (x <= 0 || y <= 0)
    return 0;
  int r = 0;
  while (x == y && x > 0)
  {
    while (y > 0)
    {
      x = x - 1;
      y = y - 1;
      r = r + 2; // non-unit step — refinement must NOT fire
    }
  }
  return r;
}
