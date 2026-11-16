int main()
{
  unsigned x = 0;
  x = x - 1;
  __ESBMC_assert(x < 5, "0 - 1 is -1 under --ir");
  return 0;
}
