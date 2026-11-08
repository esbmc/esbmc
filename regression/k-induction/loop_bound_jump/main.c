unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 200);
  unsigned i = 0, s = 0;
  while (i < n)
  {
    i++;
    s += 2;
  }
  __ESBMC_assert(s == 2 * n, "sum");
}
