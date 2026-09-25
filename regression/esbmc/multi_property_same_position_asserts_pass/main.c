/* Two bound checks share a description and a position; each needs its own
   row (esbmc/esbmc#7900). */
int a[4];
int main()
{
  unsigned i = nondet_uint(), j = nondet_uint();
  __ESBMC_assume(i < 4 && j < 4);
  int s = a[i] + a[j];
  return s;
}
