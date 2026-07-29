/* github_6484_frame_zero_extent_fail:
 * A zero stated extent makes the Phase 2B witness range empty. If that range
 * is emitted as an ASSUME it is ASSUME(false), which sits before the call and
 * discharges every later assertion -- including the unconditional assert below
 * and the out-of-bounds write -- reporting SUCCESSFUL for a function that
 * cannot execute a single valid statement.
 */
void f(int *a, int i)
{
  __ESBMC_requires(__ESBMC_is_fresh(a, 0));
  __ESBMC_requires(i == 0);
  __ESBMC_assigns(a[i]);
  __ESBMC_ensures(1);
  a[i] = 1;
  __ESBMC_assert(0, "must be reachable");
}

int main()
{
  return 0;
}
