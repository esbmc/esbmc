/* Tests for sub-expressions within the forall/exists body that are
 * themselves || or && containing a nested quantifier.
 *
 * These cover two bugs fixed in remove_sideeffects():
 *
 * Bug 1: a disjunct/conjunct that is itself an && or || containing a nested
 * quantifier would be ITE-converted, hiding the bound variable in a temp
 * symbol and producing an incorrect SMT formula.  Fixed by processing the
 * body tree recursively without ITE-converting any || or && sub-expression.
 *
 * Bug 2: an explicit (int) cast on the body creates a double typecast layer
 * ((int)(b>0||forall...) → _Bool), so the single-level typecast peel-through
 * missed the || underneath.  Fixed by peeling typecasts in a loop.
 */
int main()
{
  int a1, b1, c1;

  /* Reviewer case 1: explicit (int) cast on body — exercises double-typecast
   * peel-through.  forall b: (b>0)||forall(a,a==b); at b=0: both false.
   * => false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1, (int)((b1 > 0) || __ESBMC_forall(&a1, a1 == b1))) == 0,
    "forall b: (int)((b>0)||forall(a,a==b)) should be false");

  /* Reviewer case 2: exists with nested || inside &&.
   * exists b: (b==5) && ((b!=5) || forall(a,a==b))
   * At b=5: b!=5=false, forall(a,a==5)=false => false && false = false.
   * => exists=false => == 0 */
  __ESBMC_assert(
    __ESBMC_exists(
      &b1,
      (b1 == 5) && ((b1 != 5) || __ESBMC_forall(&a1, a1 == b1))) == 0,
    "exists b: (b==5)&&((b!=5)||forall(a,a==b)) should be false");

  /* Sub-&& with always-true inner quantifier — the case that exposed the
   * original ITE-conversion bug (b1@0 > 0 made t_new=true vacuously).
   * forall b: (b>0 && exists(c,c>0)) || (b==0)
   * exists(c,c>0)=true; (b>0&&true)||(b==0) — at b=-1: false.
   * => forall=false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1, (b1 > 0 && __ESBMC_exists(&c1, c1 > 0)) || (b1 == 0)) == 0,
    "forall b: (b>0&&exists(c,c>0))||(b==0) should be false");

  /* Disjunct is (b>0 && forall(c,c==0)) — sub-&& has inner forall, b in cond.
   * forall(c,c==0) always false => conjunction false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1,
      (b1 > 0 && __ESBMC_forall(&c1, c1 == 0)) ||
        __ESBMC_forall(&a1, a1 == b1)) == 0,
    "forall b: (b>0&&forall(c,c==0))||forall(a,a==b) should be false");

  /* Nested forall in the right operand of a sub-&&.
   * forall b: (b==17) || ((b<0) && forall(a,a==b))
   * At b=0: b==17=false; b<0=false => == 0 */
  __ESBMC_assert(
    __ESBMC_forall(
      &b1,
      (b1 == 17) || (b1 < 0 && __ESBMC_forall(&a1, a1 == b1))) == 0,
    "forall b: (b==17)||(b<0&&forall(a,a==b)) should be false");

  return 0;
}
