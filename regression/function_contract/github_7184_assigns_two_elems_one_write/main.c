/* Writing one of the two granted elements is still inside the frame. This
 * shape failed even though the body touched a single named index, because the
 * other element's check excused only the index it was built from (#7184). */
void set_first(int *arr, int n1, int n2)
{
  __ESBMC_requires(__ESBMC_is_fresh(arr, 5 * sizeof(int)));
  __ESBMC_requires(n1 >= 0 && n1 < 5);
  __ESBMC_requires(n2 >= 0 && n2 < 5);
  __ESBMC_assigns(arr[n1], arr[n2]);
  __ESBMC_ensures(1);
  arr[n1] = 7;
}

int main()
{
  return 0;
}
