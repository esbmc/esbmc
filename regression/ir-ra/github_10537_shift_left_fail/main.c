int main()
{
  /* Under --ir, 1 << 31 is 2^31, not INT_MIN. */
  int x = 1;
  x = x << 31;
  __ESBMC_assert(x < 0, "x < 0");
  return 0;
}
