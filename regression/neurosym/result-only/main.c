int main()
{
  unsigned x = nondet_uint() % 10;
  __ESBMC_assert(x < 5, "x less than five");
  return 0;
}
