/* Regression: GitHub #10642. reach_error is an assert(0), so it never holds
 * on any path; the invariant admits states that avoid it (x == 0), so its
 * counterexample is against the havoc, not a reachable state. */
extern void reach_error(void);
extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = 0, y = 0;
  __ESBMC_loop_invariant(0 <= x && x <= 1 && 0 <= y && y <= 1);
  while (__VERIFIER_nondet_int())
  {
    if (x == 1)
      y = 1;
    if (y == 1)
      reach_error();
    x = 0;
  }
  return 0;
}
