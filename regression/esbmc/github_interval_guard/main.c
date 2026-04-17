int main()
{
  int idx;
  __ESBMC_assume(0 <= idx);
  while (idx < 5)
  {
    ++idx;
  }
  __ESBMC_assert(idx >= 5, "assert");
  return 0;
}
