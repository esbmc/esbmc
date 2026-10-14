/* Instances of a violated claim after its ASSERT became a SKIP must keep its
   row (esbmc/esbmc#7900). */
int a[4];
int main()
{
  int s = 0;
  for (unsigned k = 0; k < 3; ++k)
  {
    unsigned i = nondet_uint(), j = nondet_uint();
    __ESBMC_assume(i < 4);
    if (k == 1)
      __ESBMC_assume(j < 4);
    s += a[i] + a[j];
  }
  return s;
}
