int main()
{
  int a[5];
  unsigned n = sizeof(a) / sizeof(a[0]);
  int steps = 0;
  for (unsigned i = 0; i < n; i++)
    steps++;
  __ESBMC_assert(steps == 5, "the array has five elements");
  return 0;
}
