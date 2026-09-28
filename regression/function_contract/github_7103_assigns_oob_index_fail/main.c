/* An index outside the array excuses nothing. Fed straight to a store it did:
 * the solver's index domain is only as wide as the extent, so 1024 wrapped to
 * 0 and spared the very element this body writes. The element-wise path always
 * rejected this shape, so crossing the cap turned a rejection into acceptance. */
int global[1000];

void f(int v)
{
  __ESBMC_assigns(global[1024]);
  __ESBMC_ensures(1);
  global[0] = v;
}

int main()
{
  return 0;
}
