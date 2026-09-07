/* Boundary twin of path_guard_subsumption: the caller only rules out
 * x > 0, so the callee's `v < 0` is NOT implied — x == 0 reaches the
 * other branch and the assertion is violable. Subsumption must match
 * only a condition the path guard actually decides, never a merely
 * similar one; over-eager matching would prune the feasible x == 0
 * case and turn this into a vacuous SUCCESSFUL. */
int nondet_int(void);

static int check(int v)
{
  if (v < 0)
    return 1;
  return 2;
}

int main(void)
{
  int x = nondet_int();
  if (x > 0)
    return 0;
  __ESBMC_assert(check(x) == 1, "x == 0 must stay reachable");
  return 0;
}
