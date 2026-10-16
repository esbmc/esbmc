int nondet_int();

/* C11 6.5.5p6: / truncates toward zero and % takes the dividend's sign. */
int main(void)
{
  int a = nondet_int(), b = nondet_int();
  __ESBMC_assume((a == 7 || a == -7) && (b == 2 || b == -2));
  int q = (a < 0) == (b < 0) ? 3 : -3;
  int r = a < 0 ? -1 : 1;
  __ESBMC_assert(a / b == q, "quotient truncates toward zero");
  __ESBMC_assert(a % b == r, "remainder takes the dividend's sign");
  __ESBMC_assert(a == (a / b) * b + a % b, "C11 6.5.5p6 identity");
  unsigned u = nondet_int();
  __ESBMC_assume(u == 7u);
  __ESBMC_assert(u / 2u == 3u && u % 2u == 1u, "unsigned");
  return 0;
}
