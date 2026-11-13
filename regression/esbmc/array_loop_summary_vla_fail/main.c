// A symbolic size reaches the solver as a quantified definition of the
// whole array.
unsigned nondet_uint(void);
int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 77 && n < 100);
  int a[n];
  for (unsigned i = 0; i < n; i++)
    a[i] = i;
  for (unsigned x = 0; x < n; x++)
    __ESBMC_assert(a[x] != 77, "a[x] != 77");
}
