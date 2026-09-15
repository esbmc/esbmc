/* C11 6.3.1.4p1 constrains the truncated integral part, not the value, so a
   magnitude below one always converts and MIN - 0.5 converts to MIN. The first
   revision of the #7572 check used a closed lower bound and reported both as
   undefined; that cost an SV-COMP incorrect-false on float-benchs/bary_diverge,
   where a nondet float assumed into [-1, 1] is cast to an enum.
   clang -fsanitize=float-cast-overflow accepts every conversion below. */
extern float nondet_float(void);
extern double nondet_double(void);

int main(void)
{
  float a = nondet_float();
  __ESBMC_assume(a > -1.0f && a <= 0.0f);
  unsigned int u = (unsigned int)a;
  __ESBMC_assert(u == 0u, "a magnitude below one truncates to zero");

  double d = nondet_double();
  __ESBMC_assume(d == -2147483648.5);
  int i = (int)d;
  __ESBMC_assert(i == -2147483647 - 1, "INT_MIN - 0.5 truncates to INT_MIN");
  return 0;
}
