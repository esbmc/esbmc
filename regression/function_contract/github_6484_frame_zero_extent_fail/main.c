/* github_6484_frame_zero_extent_fail:
 * A zero stated extent makes the Phase 2B witness range empty. If that range
 * is emitted as an ASSUME it is ASSUME(false), which sits before the call and
 * discharges every assertion after it, so the out-of-bounds write below is
 * reported as SUCCESSFUL. Clamping instead leaves the write reachable, and it
 * fails its bounds check.
 */
void f(int *a, int i)
{
  __ESBMC_requires(__ESBMC_is_fresh(a, 0));
  __ESBMC_requires(i == 0);
  __ESBMC_assigns(a[i]);
  __ESBMC_ensures(1);
  a[i] = 1;
}

int main()
{
  return 0;
}
