// GitHub #6154: a callee that writes through a pointer is NOT summarizable --
// the summarizer models only local scalars, so accepting this would drop the
// store.  It must be refused (and the quantifier then diagnosed) rather than
// silently turned into a pure expression.  __ESBMC_exists has no skolemization
// fallback, so refusal surfaces as the diagnostic below.
int store(int a, int *p)
{
  *p = a;
  return a;
}

int main()
{
  int q = 0;
  int v;
  __ESBMC_assert(__ESBMC_exists(&v, store(v, &q) == 7), "not summarizable");
  return 0;
}
