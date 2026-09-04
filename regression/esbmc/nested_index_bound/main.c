int main()
{
  int a[2][3] = {{1, 1, 1}, {1, 1, 1}};
  int sum = 0;
  for (int *p = &a[0][0]; p != &a[1][2]; p++)
    sum += *p;
  __ESBMC_assert(sum == 5, "flat walk covers five elements");
  return 0;
}
