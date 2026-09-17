/* github #7184: excusing every declared index must not excuse the ones the
 * clause did not name. arr[2] is outside the frame. */
void array_swap(int *arr, int n, int n1, int n2)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, n * sizeof(int)));
  __ESBMC_requires(n >= 3);
  __ESBMC_requires(n1 == 0 && n2 == 1);
  __ESBMC_assigns(arr[n1], arr[n2]);

  int temp = arr[n1];
  arr[n1] = arr[n2];
  arr[n2] = temp;
  arr[2] = 7;
}

int main()
{
  return 0;
}
