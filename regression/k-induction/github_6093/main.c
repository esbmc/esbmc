int main()
{
  int idx, max, res;

  __ESBMC_assume(0 <= max);
  res = 0;
  idx = 0;

  while (idx < max)
  {
    ++idx;
    ++res;
  }
  __ESBMC_assert(idx == max, "A.1");
}
