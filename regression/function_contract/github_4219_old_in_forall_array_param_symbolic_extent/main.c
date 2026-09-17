// The sibling github_4219_old_in_forall_array_param test uses a compile-time
// #define N for r's extent (N * sizeof(int)), so the region snapshot's
// element count folds to a constant and the copy loop needs no real
// unwinding. This variant uses a runtime parameter for both the extent and
// the constant added, so the element count is fully symbolic and the copy
// loop genuinely participates in --unwind bookkeeping (#7057).
#define BOUND 100

void increment_array_by(int *arr, int n, int c)
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 0 && n <= 5);
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < n) || (arr[j] > -BOUND && arr[j] < BOUND)));
  __ESBMC_ensures(__ESBMC_forall(
    &j, !(j < n) || (arr[j] == __ESBMC_old(arr[j]) + c)));
  __ESBMC_assigns(arr);

  for (int i = 0; i < n; i++)
    arr[i] = arr[i] + c;
}

int main(void)
{
  return 0;
}
