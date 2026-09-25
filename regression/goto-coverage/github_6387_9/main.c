void helper(int x)
{
  __ESBMC_assert(x != 7, "helper never sees 7");
}
void target(int y)
{
  __ESBMC_assert(y == y, "target trivial");
  helper(y);
}
int main()
{
  int z;
  target(z);
  return 0;
}
