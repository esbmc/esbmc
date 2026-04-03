// OK: idx declared after max initialization; should PASS (issue #3939)
int main()
{
  int idx_v0;
  int vec[3];
  int max;
  max = vec[0];
  int idx;
  __ESBMC_loop_invariant(
    __ESBMC_forall(
      &idx_v0,
      !(0 <= idx_v0) ||
      !(idx_v0 < 3) ||
      !(idx_v0 < idx) ||
      (vec[idx_v0] <= max)));
  __ESBMC_loop_invariant(1 <= idx);
  for (idx = 1; idx < 3; ++idx)
  {
    if (max < vec[idx])
      max = vec[idx];
  }
  return 0;
}
