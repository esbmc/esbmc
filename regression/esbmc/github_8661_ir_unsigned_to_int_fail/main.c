unsigned nondet_uint();
int main(void)
{
  unsigned u = nondet_uint();
  __ESBMC_assume(u == 3000000000u);
  int x = (int)u;
  return x;
}
