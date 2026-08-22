/* github #7184: a clause naming two elements of the same array got one witness
 * assertion per target, each excusing only its own index, so the legal write to
 * arr[n2] tripped the arr[n1] witness. */
void array_swap(int *arr, int n, int n1, int n2)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 1);
  __ESBMC_requires(n1 >= 0 && n1 < n);
  __ESBMC_requires(n2 >= 0 && n2 < n);
  __ESBMC_ensures(arr[n1] == __ESBMC_old(arr[n2]));
  __ESBMC_ensures(arr[n2] == __ESBMC_old(arr[n1]));
  __ESBMC_assigns(arr[n1], arr[n2]);

  int temp = arr[n1];
  arr[n1] = arr[n2];
  arr[n2] = temp;
}

int main()
{
  return 0;
}
