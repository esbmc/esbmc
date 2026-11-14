int main()
{
  int x = -2147483647 - 1;
  x = -x;
  __ESBMC_assert(x > 0, "-INT_MIN is 2^31 under --ir");
  return 0;
}
