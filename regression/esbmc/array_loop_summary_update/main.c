// A store at a symbolic index after the loop leaves the read unreduced, so
// the solver gets the whole summarised array.
unsigned nondet_uint(void);
int a[100];
int main()
{
  for (unsigned i = 0; i < 100; i++)
    a[i] = i;
  unsigned c = nondet_uint();
  __ESBMC_assume(c < 100);
  a[c] = -1;
  unsigned y = nondet_uint();
  __ESBMC_assume(y < 100);
  __ESBMC_assert(a[y] == y || y == c, "only a[c] changed");
}
