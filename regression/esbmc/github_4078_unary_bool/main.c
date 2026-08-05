// C11 6.5.3.3: the operand of unary - and ~ undergoes integer promotion, so a
// boolean one -- a comparison, || or && -- becomes int. The frontend left it
// boolean, and the solver was then handed a boolean where it wanted a
// bitvector: z3 reported "operator is applied to arguments of the wrong sort"
// for ~ and "ast is not an expression" for -, two of the signatures listed in
// issue #4078. Reduced from gcc.c-torture/execute/pr112581-1.c.
int a, b;

int main(void)
{
  __ESBMC_assert(~(a || b) == -1, "~0 is -1");
  __ESBMC_assert(-(a || b) == 0, "-0 is 0");

  __ESBMC_assert(~(1 || b) == -2, "~1 is -2");
  __ESBMC_assert(-(1 || b) == -1, "-1 is -1");

  __ESBMC_assert(~(a && b) == -1, "&& promotes too");
  __ESBMC_assert(~(a < 1) == -2, "a comparison promotes too");

  // The binary path already inserted the cast; keep it pinned alongside.
  __ESBMC_assert((a || b) + 1 == 1, "binary operands were never affected");
  return 0;
}
