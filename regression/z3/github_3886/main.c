/* Regression test for https://github.com/esbmc/esbmc/discussions/3886
 * Nested __ESBMC_forall quantifiers with integer comparisons.
 *
 * All assertions below must pass (VERIFICATION SUCCESSFUL).
 *
 * Single-level quantifiers (sanity checks):
 *   forall a: (a == 17)  => false => == 0
 *   forall a: (a > 17)   => false => == 0
 *
 * Nested quantifiers - passing cases with cross-variable comparisons:
 *   forall a b: (a > b) || (b > a)    => false (a==b==0) => == 0
 *   forall a b: (a == b) || (b == 17) => false (a==0, b==0) => == 0
 *   forall a b: (a == b) || (b > 17)  => false (a==0, b==0) => == 0
 *   forall a b: (a > b) || (b == 17)  => false (a==0, b==0) => == 0
 *
 * Nested quantifiers - previously broken cases (the bug):
 *   forall a b: (a > b) || (b > 17)   => false (a==0, b==0) => == 0
 *   forall a b: (a == 17) || (b == 17) => false (a==0, b==0) => == 0
 */
int main()
{
  int a1, b1;

  __ESBMC_assert(
    __ESBMC_forall(&a1, a1 == 17) == 0,
    "forall a: a==17 should be false");

  __ESBMC_assert(
    __ESBMC_forall(&a1, a1 > 17) == 0,
    "forall a: a>17 should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 > b1) || (b1 > a1))) == 0,
    "forall a b: (a>b)||(b>a) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 == b1) || (b1 == 17))) == 0,
    "forall a b: (a==b)||(b==17) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 == b1) || (b1 > 17))) == 0,
    "forall a b: (a==b)||(b>17) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 > b1) || (b1 == 17))) == 0,
    "forall a b: (a>b)||(b==17) should be false");

  /* Previously broken: outer variable only in one disjunct */
  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 > b1) || (b1 > 17))) == 0,
    "forall a b: (a>b)||(b>17) should be false");

  /* Previously broken: each disjunct uses only one quantified variable */
  __ESBMC_assert(
    __ESBMC_forall(
      &a1, __ESBMC_forall(&b1, (a1 == 17) || (b1 == 17))) == 0,
    "forall a b: (a==17)||(b==17) should be false");

  return 0;
}
