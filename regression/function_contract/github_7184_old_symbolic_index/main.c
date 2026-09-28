/* github #7184: the __ESBMC_old(arr[i]) snapshot was taken before the requires
 * clause was assumed, so it dereferenced at an unconstrained index and reported
 * "array bounds violated" on a contract that holds. */
void array_bump(int *arr, int n, int i)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 1);
  __ESBMC_requires(i >= 0 && i < n);
  __ESBMC_ensures(arr[i] == __ESBMC_old(arr[i]) + 1);
  __ESBMC_assigns(arr[i]);

  arr[i] = arr[i] + 1;
}

int main()
{
  return 0;
}
