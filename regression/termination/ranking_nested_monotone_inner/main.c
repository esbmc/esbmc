// Pins the monotone-counter inner-loop post-condition refinement.
//
// The outer's continuation requires *x == *y && *x > 0*. The inner is
// a single-path body where every modified scalar has a unit-step
// assign (y--, x--), and the anchor y has step -1 with guard `y > 0`.
//
// Pure-havoc summary would make post-inner *x fresh nondet → no
// measure on *x decreases pre→post-outer → ranking fails. The
// refinement emits post-inner *y == 0 and *x == pre_x - pre_y,
// which combined with the outer's `*x == *y && *x > 0` precondition
// forces a contradiction after one iteration — ranking discharges.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  if (x <= 0 || y <= 0)
    return 0;
  while (x == y && x > 0)
  {
    while (y > 0)
    {
      x = x - 1;
      y = y - 1;
    }
  }
  return 0;
}
