/* Sparing the named element of a pointer array must not spare the array: the
 * report has to name the element written, not the array. */
int *pa[10];
int a;

void f(int i, int *v)
{
  __ESBMC_requires(i >= 1 && i < 10);
  __ESBMC_assigns(pa[i]);
  __ESBMC_ensures(1);
  pa[i] = v;
  pa[0] = &a; /* not in __ESBMC_assigns */
}

int main()
{
  return 0;
}
