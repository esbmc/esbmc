/* Regression test for nested __ESBMC_forall - three-level nesting.
 * Tests that expand_quantifier_defs_in recurses through two levels of
 * forall_defs_ expansion (inner body inlined into middle, then middle
 * inlined into outer).
 *
 * All assertions below must pass (VERIFICATION SUCCESSFUL).
 *
 *   forall a b c: (a==17)||(b==17)||(c==17)  => false (a=b=c=0) => == 0
 *   forall a b c: (a>b)||(b>c)||(c>a)        => false (a=b=c=0) => == 0
 *   forall a b c: (a==b)&&(b==c)             => false (b=0,c=1 makes b!=c) => == 0
 *   forall a b c: (a==a)&&(b==b)&&(c==c)     => true (tautology) => == 1
 */
int main()
{
  int a1, b1, c1;

  __ESBMC_assert(
    __ESBMC_forall(
      &a1,
      __ESBMC_forall(
        &b1,
        __ESBMC_forall(&c1, (a1 == 17) || (b1 == 17) || (c1 == 17)))) == 0,
    "forall a b c: (a==17)||(b==17)||(c==17) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1,
      __ESBMC_forall(
        &b1,
        __ESBMC_forall(&c1, (a1 > b1) || (b1 > c1) || (c1 > a1)))) == 0,
    "forall a b c: (a>b)||(b>c)||(c>a) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1,
      __ESBMC_forall(
        &b1, __ESBMC_forall(&c1, (a1 == b1) && (b1 == c1)))) == 0,
    "forall a b c: (a==b)&&(b==c) should be false");

  __ESBMC_assert(
    __ESBMC_forall(
      &a1,
      __ESBMC_forall(
        &b1,
        __ESBMC_forall(&c1, (a1 == a1) && (b1 == b1) && (c1 == c1)))) == 1,
    "forall a b c: (a==a)&&(b==b)&&(c==c) should be true");

  return 0;
}
