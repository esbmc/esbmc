/* github #7184: the indices one clause names need not share a type, so the
 * grouped frame check stays live for them: arr[2] is outside the frame. */
void f(int *arr, int n, int i, unsigned long k)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 3);
  __ESBMC_requires(i == 0);
  __ESBMC_requires(k == 1);
  __ESBMC_assigns(arr[i], arr[k]);

  arr[i] = 1;
  arr[k] = 2;
  arr[2] = 3;
}

int main()
{
  return 0;
}
