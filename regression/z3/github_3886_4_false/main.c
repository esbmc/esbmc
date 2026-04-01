/* Negative regression test: assertion must FAIL.
 * forall b: (b>0) || forall(a, a>0)
 * At b=0: b>0 is false; forall a: a>0 is false (a=0) => false.
 * So the outer forall is false, not 1 => assertion fails. */
int main()
{
  int a1, b1;

  __ESBMC_assert(
    __ESBMC_forall(&b1, (b1 > 0) || __ESBMC_forall(&a1, a1 > 0)) == 1,
    "forall b: (b>0)||forall(a,a>0) should be false — assertion must fail");

  return 0;
}
