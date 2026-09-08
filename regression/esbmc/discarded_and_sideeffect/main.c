/* The && arm of the same lowering: `A && f()` with the value discarded is the
 * statement `if (A) f();`, so the call is reached exactly when A holds. */
int calls;
int bump(void)
{
  calls++;
  return 1;
}
int main(void)
{
  unsigned n;
  __ESBMC_assume(n < 4);
  (void)((n > 2) && bump());
  __ESBMC_assert(calls == (n > 2 ? 1 : 0), "bump runs iff n > 2");
  return 0;
}
