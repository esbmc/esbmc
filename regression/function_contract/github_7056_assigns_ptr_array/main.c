/* An element of an array *of pointers* is pointer-typed, so it used to be
 * classed as a pointer target and the whole array was still asserted
 * unchanged -- failing even here, where the body writes only the element the
 * clause names. */
int *pa[10];

void f(int i, int *v)
{
  __ESBMC_requires(i >= 0 && i < 10);
  __ESBMC_assigns(pa[i]);
  __ESBMC_ensures(1);
  pa[i] = v;
}

int main()
{
  return 0;
}
