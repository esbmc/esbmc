/* Widening the excusal to every named index must not excuse an index the
 * clause never named: this body writes a third element and must be caught. */
void set_third(int *arr, int n1, int n2, int n3)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, 5 * sizeof(int)));
  __ESBMC_requires(n1 >= 0 && n1 < 5);
  __ESBMC_requires(n2 >= 0 && n2 < 5);
  __ESBMC_requires(n3 >= 0 && n3 < 5);
  __ESBMC_requires(n3 != n1 && n3 != n2);
  __ESBMC_assigns(arr[n1], arr[n2]);
  __ESBMC_ensures(1);
  arr[n3] = 7;
}

int main()
{
  return 0;
}
