int main()
{
  int x = 65536;
  x = x * 65536;
  __ESBMC_assert(x != 0, "2^16 * 2^16 is 2^32 under --ir");
  return 0;
}
