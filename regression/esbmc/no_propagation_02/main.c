int main()
{
  int x = 42;
  int y = x + 1;
  __ESBMC_assert(y == 43, "y is 43");
  return 0;
}
