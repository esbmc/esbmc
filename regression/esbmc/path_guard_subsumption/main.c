/* A callee's re-check of a condition the caller's path already decided
 * must be resolved from the path guard, not re-forked: without
 * path-guard subsumption the contradictory `v >= 0` branch below is
 * explored symbolically, and everything it dominates is dragged into
 * the SSA for nothing. With it, `check` collapses on this path and the
 * assertion simplifies away: 0 VCCs remain. */
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
  if (x >= 0)
    return 0;
  __ESBMC_assert(check(x) == 1, "negative path decided by subsumption");
  return 0;
}
