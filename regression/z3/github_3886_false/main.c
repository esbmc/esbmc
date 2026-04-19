/* Regression test for nested __ESBMC_forall - VERIFICATION FAILED path.
 * Purpose: confirm that a false nested forall is not silently accepted.
 * The nested forall evaluates to false (a=b=0 satisfies neither a>b nor
 * b>a), so asserting it true must cause VERIFICATION FAILED.
 *
 *   forall a b: (a > b) || (b > a)  =>  false  (a=b=0 is counterexample)
 */
int main()
{
  int a1, b1;

  __ESBMC_assert(
    __ESBMC_forall(&a1, __ESBMC_forall(&b1, (a1 > b1) || (b1 > a1))),
    "nested forall (a>b)||(b>a) incorrectly asserted true");

  return 0;
}
