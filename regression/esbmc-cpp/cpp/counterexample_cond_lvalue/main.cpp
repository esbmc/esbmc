// The holding counterpart of counterexample_cond_lvalue_fail.
extern "C" bool nondet_bool();

int a, b;

int main()
{
  bool c = nondet_bool();
  (c ? a : b) = 5;
  __ESBMC_assert(a == 5 || b == 5, "neither is 5");
  return 0;
}
