// Regression test for GitHub #3995: nested forall/exists with array accesses
// should not produce spurious array bounds violations.
int main()
{
  int x;
  const int SIZE = 5;
  int i_vec[SIZE] = {x, x, x, x, x};
  int o_vec[SIZE];
  int i1_idx;
  int iv_idx, ov_idx;

  for (i1_idx = 0; i1_idx < SIZE; ++i1_idx)
    o_vec[i1_idx] = i_vec[i1_idx];

  // Assert 1: exists without outer forall (baseline)
  __ESBMC_assert(
    !(0 <= ov_idx && ov_idx < SIZE) ||
      __ESBMC_exists(
        &iv_idx,
        (0 <= iv_idx) && (iv_idx < SIZE) && (o_vec[ov_idx] == i_vec[iv_idx])),
    "assert 1");

  // Assert 2: forall wrapping exists, bounds checks only (no array access)
  __ESBMC_assert(
    __ESBMC_forall(
      &ov_idx,
      !(0 <= ov_idx && ov_idx < SIZE) ||
        __ESBMC_exists(
          &iv_idx,
          (0 <= iv_idx) && (iv_idx < SIZE) && (0 <= ov_idx) &&
            (ov_idx < SIZE))),
    "assert 2");

  // Assert 3: forall wrapping exists with array accesses (the bug)
  __ESBMC_assert(
    __ESBMC_forall(
      &ov_idx,
      !(0 <= ov_idx && ov_idx < SIZE) ||
        __ESBMC_exists(
          &iv_idx,
          (0 <= iv_idx) && (iv_idx < SIZE) &&
            (o_vec[ov_idx] == i_vec[iv_idx]))),
    "assert 3");

  return 0;
}
