/* github #7184: a requires clause that itself reads __ESBMC_old cannot be
 * assumed before its own snapshots — the ensures must still be checked. */
void bump(int *arr, int n, int i)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(__ESBMC_old(n) >= 1);
  __ESBMC_requires(i == 0);
  __ESBMC_ensures(arr[0] == 7);
  __ESBMC_assigns(arr[i]);

  arr[i] = 8;
}

int main()
{
  return 0;
}
