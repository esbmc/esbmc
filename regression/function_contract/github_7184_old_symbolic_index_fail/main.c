/* github #7184: companion to github_7184_old_symbolic_index — assuming the
 * requires clause before the snapshot must not stop the ensures failing. */
void array_bump(int *arr, int n, int i)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 1);
  __ESBMC_requires(i >= 0 && i < n);
  __ESBMC_ensures(arr[i] == __ESBMC_old(arr[i]) + 1);
  __ESBMC_assigns(arr[i]);

  arr[i] = arr[i] + 2;
}

int main()
{
  return 0;
}
