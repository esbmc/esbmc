// A "__CPROVER_uninterpreted_*" function (CBMC's convention) is an arbitrary
// but fixed function of its arguments. ESBMC maps it onto the same modelling as
// the native __ESBMC_uninterpreted_* prefix: any body is ignored and functional
// congruence is enforced, so calling it twice with equal arguments must yield
// the same result.
int __CPROVER_uninterpreted_f(int);

int main()
{
  int x;
  int a = __CPROVER_uninterpreted_f(x);
  int b = __CPROVER_uninterpreted_f(x);
  __ESBMC_assert(a == b, "congruence: equal arguments give equal results");
  return 0;
}
