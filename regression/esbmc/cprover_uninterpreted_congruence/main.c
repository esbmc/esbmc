// A "__CPROVER_uninterpreted_*" function is an arbitrary but fixed function of
// its arguments. With --cprover-uninterpreted-functions ESBMC ignores any body
// and enforces functional congruence: calling it twice with equal arguments
// must yield the same result. Without the flag each bodyless call is an
// independent nondeterministic value, so this assertion would not hold.
int __CPROVER_uninterpreted_f(int);

int main()
{
  int x;
  int a = __CPROVER_uninterpreted_f(x);
  int b = __CPROVER_uninterpreted_f(x);
  __ESBMC_assert(a == b, "congruence: equal arguments give equal results");
  return 0;
}
