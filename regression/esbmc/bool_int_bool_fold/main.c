// Widening a _Bool yields exactly 0 or 1, so comparing it against zero
// recovers the bool. The simplifier folds that round trip away; these
// assertions pin the semantics it must preserve while doing so. #4626.
_Bool nondet_bool(void);
int nondet_int(void);

int main(void)
{
  _Bool b = nondet_bool();

  __ESBMC_assert(((int)b != 0) == b, "(int)b != 0 is b");
  __ESBMC_assert(((int)b == 0) == !b, "(int)b == 0 is !b");
  __ESBMC_assert((_Bool)(int)b == b, "(_Bool)(int)b is b");

  if ((int)b != 0)
    __ESBMC_assert(b, "b holds on the taken branch");
  else
    __ESBMC_assert(!b, "b fails on the other branch");

  // The reverse trip keeps only whether i was non-zero, so it is not the
  // identity on i. Folding it away would be unsound.
  int i = nondet_int();
  __ESBMC_assume(i == 2);
  __ESBMC_assert((int)(_Bool)i == 1, "(int)(_Bool)2 is 1, not 2");

  return 0;
}
