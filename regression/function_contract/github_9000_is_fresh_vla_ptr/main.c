/* #9000: a pointer that may be a VLA large enough or NULL; the VLA is sized by
 * its allocation. */
_Bool nondet_bool();
unsigned nondet_uint();
void g(int *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, 4 * sizeof(int)));
  __ESBMC_ensures(1);
  __ESBMC_assigns();
}
int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n >= 4 && n < 8);
  int v[n];
  int *p = nondet_bool() ? v : 0;
  __ESBMC_assume(p != 0);
  g(p);
  return 0;
}
