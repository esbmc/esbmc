/* Tests for __ESBMC_forall/__ESBMC_exists bodies that contain both a
 * nested quantifier call and a nondet function value.
 *
 * These verify that remove_sideeffects() correctly lifts the nested
 * quantifier call to a temp while leaving the nondet-derived expression
 * in place, and that the outer quantifier body is then encoded correctly.
 *
 * __ESBMC_assume constrains nondet values so each assertion is deterministic.
 */
int main()
{
  int a1, b1;
  int n;

  /* n > 0: first disjunct (n>0) is true for all b, so body always true.
   * forall b: (n>0) || forall(a, a==b) => true => == 1 */
  n = nondet_int();
  __ESBMC_assume(n > 0);
  __ESBMC_assert(
    __ESBMC_forall(&b1, (n > 0) || __ESBMC_forall(&a1, a1 == b1)) == 1,
    "forall b: (n>0)||forall(a,a==b) should be true when n>0");

  /* n == 0: first disjunct false; forall a: a==b always false.
   * forall b: false || false => false => == 0 */
  n = nondet_int();
  __ESBMC_assume(n == 0);
  __ESBMC_assert(
    __ESBMC_forall(&b1, (n > 0) || __ESBMC_forall(&a1, a1 == b1)) == 0,
    "forall b: (n>0)||forall(a,a==b) should be false when n==0");

  /* n > 5: 3-way || with symbolic n.  collect_or_disjuncts flattens all
   * three operands; the symbolic n appears in the first disjunct.
   * At b=1: (1>n) false for n>5; (1<=0) false; forall a: a==0 false.
   * forall b: ... = false => == 0 */
  n = nondet_int();
  __ESBMC_assume(n > 5);
  __ESBMC_assert(
    __ESBMC_forall(
      &b1, (b1 > n) || (b1 <= 0) || __ESBMC_forall(&a1, a1 == 0)) == 0,
    "forall b: (b>n)||(b<=0)||forall(a,a==0) should be false when n>5");

  /* n > 0: exists b: (b<n) && forall(a, a+n>0).
   * forall a: a+n>0 is false (pick a = -n-1 => a+n = -1).
   * exists b: (b<n) && false = false => == 0 */
  n = nondet_int();
  __ESBMC_assume(n > 0);
  __ESBMC_assert(
    __ESBMC_exists(&b1, (b1 < n) && __ESBMC_forall(&a1, a1 + n > 0)) == 0,
    "exists b: (b<n)&&forall(a,a+n>0) should be false when n>0");

  return 0;
}
