extern void __ESBMC_assume(_Bool);
extern _Bool __ESBMC_forall(void *, _Bool);
extern _Bool __ESBMC_exists(void *, _Bool);

int main()
{
  unsigned n;
  int arr[n];
  unsigned i;

  __ESBMC_assume(n > 1 && n < 10);

  int array_is_sorted =
    __ESBMC_forall(&i, !(i >= 0 && i < (n - 1)) || arr[i] <= arr[i + 1]);

  // Assume sorted order
  __ESBMC_assume(array_is_sorted);
  // Check whether at least one array is generated (prevents quantifiers-sort vacously hold)
  __ESBMC_assert(0, "reachable");
  return 0;
}
