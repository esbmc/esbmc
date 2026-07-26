int main()
{
  unsigned x = nondet_uint() % 10;
  __ESBMC_assert(x < 10, "x in range");
  return 0;
}
