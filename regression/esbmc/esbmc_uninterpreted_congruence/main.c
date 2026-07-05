// An "__ESBMC_uninterpreted_*" function is an arbitrary but fixed function of
// its arguments. It is modelled as a genuine uninterpreted function by default
// (no flag needed, unlike the __CPROVER_uninterpreted_* alias): ESBMC ignores
// any body and enforces functional congruence, so calling it twice with equal
// arguments must yield the same result.
int __ESBMC_uninterpreted_f(int);

int main()
{
  int x;
  int a = __ESBMC_uninterpreted_f(x);
  int b = __ESBMC_uninterpreted_f(x);
  __ESBMC_assert(a == b, "congruence: equal arguments give equal results");
  return 0;
}
