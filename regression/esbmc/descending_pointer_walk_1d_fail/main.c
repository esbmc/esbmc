int main()
{
  int a[5] = {1, 1, 1, 1, 1};
  int sum = 0;
  for (int *p = &a[4]; p != &a[0]; p--)
    sum += *(p - 1);
  __ESBMC_assert(sum == 3, "the walk does not cover three elements");
  return 0;
}
