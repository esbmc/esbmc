/* Regression test for nested __ESBMC_forall/__ESBMC_exists where the outer
 * quantifier body is a || or && expression containing the nested quantifier
 * call as a direct operand.
 *
 * Previously, remove_sideeffects() converted the || / && body into a
 * short-circuit ITE chain, replacing it with a single temp symbol t.
 * The outer forall became forall(b, t) where b no longer appeared in t,
 * making replace_name_in_body() unable to substitute the bound variable.
 */
int main()
{
  int a1, b1;

  /* forall b: (b==0) || forall(a, a==0)
   * At b=1: LHS false; inner forall false (a can be 1) => false */
  __ESBMC_assert(
    __ESBMC_forall(&b1, (b1 == 0) || __ESBMC_forall(&a1, a1 == 0)) == 0,
    "forall b: (b==0)||forall(a,a==0) should be false");

  /* forall b: forall(a, a==b) || (b==17)
   * At b=0: inner forall false; b==17 false => false */
  __ESBMC_assert(
    __ESBMC_forall(&b1, __ESBMC_forall(&a1, a1 == b1) || (b1 == 17)) == 0,
    "forall b: forall(a,a==b)||(b==17) should be false");

  /* forall b: (b>=0) || exists(a, a<0)
   * exists a: a<0 is true => RHS always true => true */
  __ESBMC_assert(
    __ESBMC_forall(&b1, (b1 >= 0) || __ESBMC_exists(&a1, a1 < 0)) == 1,
    "forall b: (b>=0)||exists(a,a<0) should be true");

  /* exists b: (b>0) && forall(a, a==b)
   * forall a: a==b always false => conjunction false => false */
  __ESBMC_assert(
    __ESBMC_exists(&b1, (b1 > 0) && __ESBMC_forall(&a1, a1 == b1)) == 0,
    "exists b: (b>0)&&forall(a,a==b) should be false");

  /* forall b: (b>=0 || b<0) && exists(a, a!=b)
   * LHS tautology (trichotomy); exists a: a!=b is true for any b => true */
  __ESBMC_assert(
    __ESBMC_forall(&b1, (b1 >= 0 || b1 < 0) && __ESBMC_exists(&a1, a1 != b1)) == 1,
    "forall b: (b>=0||b<0)&&exists(a,a!=b) should be true");

  return 0;
}
