/* Additional branch coverage for the __ESBMC_forall/__ESBMC_exists fix
 * in remove_sideeffects() (goto_sideeffects.cpp).
 *
 * Covers paths not exercised by github_3886_4:
 *  - collect_or_disjuncts / collect_and_conjuncts with n > 2 operands
 *  - || body where BOTH disjuncts are nested quantifier calls
 *  - && body where BOTH conjuncts are nested quantifier calls
 *  - fallthrough: body has a sideeffect but is not || or &&
 */
int main()
{
  int a1, b1, c1;

  /* 3-way || — exercises collect_or_disjuncts flattening n=3 operands.
   * forall b: (b>5) || (b<-5) || forall(a, a==0)
   * At b=0: all three are false => body false => forall false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1, (b1 > 5) || (b1 < -5) || __ESBMC_forall(&a1, a1 == 0)) == 0,
    "forall b: (b>5)||(b<-5)||forall(a,a==0) should be false");

  /* 3-way && — exercises collect_and_conjuncts flattening n=3 operands.
   * exists b: (b>0) && (b<100) && forall(a, a==b)
   * forall a: a==b is always false => conjunction false => exists false => == 0 */
  __ESBMC_assert(
    __ESBMC_exists(
      &b1, (b1 > 0) && (b1 < 100) && __ESBMC_forall(&a1, a1 == b1)) == 0,
    "exists b: (b>0)&&(b<100)&&forall(a,a==b) should be false");

  /* Both || operands are nested forall calls (both are sideeffects).
   * forall b: forall(a, a>b) || forall(c, c<b)
   * At b=0: forall a: a>0 = false; forall c: c<0 = false => false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1,
      __ESBMC_forall(&a1, a1 > b1) || __ESBMC_forall(&c1, c1 < b1)) == 0,
    "forall b: forall(a,a>b)||forall(c,c<b) should be false");

  /* Both && operands are nested forall calls (both are sideeffects).
   * exists b: forall(a, a>b) && forall(c, c<b)
   * Both inner foralls are false at b=0 => conjunction false => exists false => == 0 */
  __ESBMC_assert(
    __ESBMC_exists(
      &b1,
      __ESBMC_forall(&a1, a1 > b1) && __ESBMC_forall(&c1, c1 < b1)) == 0,
    "exists b: forall(a,a>b)&&forall(c,c<b) should be false");

  /* Fallthrough: body has a sideeffect but is NOT || or &&.
   * The new handler does not apply; remove_sideeffects falls through to
   * the generic Forall_operands path which handles it correctly.
   * forall b: !forall(a, a > b)
   * = forall b: exists a: a <= b  (pick a = b => true for all b)
   * = true => == 1 */
  __ESBMC_assert(
    __ESBMC_forall(&b1, !__ESBMC_forall(&a1, a1 > b1)) == 1,
    "forall b: !forall(a,a>b) should be true");

  return 0;
}
