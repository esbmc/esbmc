/* C11 6.5.3.3: the operand of unary - and ~ undergoes integer promotion, so a
   boolean one -- a comparison, || or && -- becomes int. The IREP2 pass handled
   only the complex unary case, so the boolean operand reached the solver where
   a bitvector was wanted and bitwuzla aborted (#4078). */
int a, b;

int main(void)
{
  __ESBMC_assert(~(a || b) == -1, "~0 is -1");
  __ESBMC_assert(-(a || b) == 0, "-0 is 0");
  __ESBMC_assert(~(1 || b) == -2, "~1 is -2");
  __ESBMC_assert(-(1 || b) == -1, "-1 is -1");
  __ESBMC_assert(~(a && b) == -1, "&& promotes too");
  __ESBMC_assert(~(a < 1) == -2, "a comparison promotes too");
  return 0;
}
