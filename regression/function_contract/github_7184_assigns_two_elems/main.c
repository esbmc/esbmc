/* A clause naming two elements of the same array grants both. Swapping them is
 * a write inside that frame, so it must verify -- and used to be rejected,
 * because each element check excused only its own index (#7184). */
void array_swap(int *arr, int n1, int n2)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, 5 * sizeof(int)));
  __ESBMC_requires(n1 >= 0 && n1 < 5);
  __ESBMC_requires(n2 >= 0 && n2 < 5);
  __ESBMC_assigns(arr[n1], arr[n2]);
  __ESBMC_ensures(1);
  int temp = arr[n1];
  arr[n1] = arr[n2];
  arr[n2] = temp;
}

int main()
{
  return 0;
}
