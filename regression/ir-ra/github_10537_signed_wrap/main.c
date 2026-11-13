int main()
{
  int x = 2147483647;
  x = x + 1;
  __ESBMC_assert(x > 0, "INT_MAX + 1 stays positive under --ir");
  return 0;
}
