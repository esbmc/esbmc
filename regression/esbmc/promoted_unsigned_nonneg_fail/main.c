/* Soundness boundary of is_provably_nonneg: each case below is
 * falsifiable and its own FAILED line is pinned, so a relaxation of
 * any guard erases that line and trips the test.
 *   1. plain signed nondet — the base case;
 *   2. SAME-width unsigned-to-signed cast: (int)u32 reinterprets the
 *      bits, the sign bit can be set (relaxing the strict `>` width
 *      check to `>=` wrongly folds this);
 *   3. NARROWING unsigned-to-signed cast: (short)u32 truncates, the
 *      kept bits can set the sign;
 *   4. widening SIGNED-to-signed cast: sign-extension preserves a
 *      negative (relaxing the unsigned-source check wrongly folds
 *      this). */
int nondet_int(void);
unsigned int nondet_u4(void);
unsigned short nondet_u2(void);
short nondet_s2(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(x >= 0, "a signed nondet can be negative");

  int same_width = (int)nondet_u4();
  __ESBMC_assert(same_width >= 0, "same-width cast keeps the sign bit");

  short narrowed = (short)nondet_u4();
  __ESBMC_assert(narrowed >= 0, "narrowing keeps the sign bit");

  int sign_extended = (int)nondet_s2();
  __ESBMC_assert(sign_extended >= 0, "sign-extension keeps a negative");
  return 0;
}
