// Congruence must not over-constrain: an uninterpreted function is NOT forced
// to be constant. With distinct arguments it may return distinct results, so
// this assertion is genuinely violable and ESBMC must report it. This guards
// against an unsound "always returns the same value" simplification. No flag is
// needed for the native __ESBMC_uninterpreted_* prefix.
int __ESBMC_uninterpreted_f(int);

int main()
{
  int x, y;
  __ESBMC_assume(x != y);
  int a = __ESBMC_uninterpreted_f(x);
  int b = __ESBMC_uninterpreted_f(y);
  __ESBMC_assert(a == b, "distinct arguments need not give equal results");
  return 0;
}
