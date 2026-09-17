int main()
{
  int a[5];
  unsigned n = sizeof(a) / sizeof(a[0]);
  int steps = 0;
  for (unsigned i = 0; i < n; i++)
    steps++;
  __ESBMC_assert(steps == 4, "the array does not have four elements");
  return 0;
}
