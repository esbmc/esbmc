unsigned nondet_uint(void);

int main()
{
  unsigned x = nondet_uint(), y = nondet_uint();
  unsigned m = x & y;
  __ESBMC_assert(m != 6, "masked");
  __ESBMC_assert(x != 2, "operand");
  return 0;
}
