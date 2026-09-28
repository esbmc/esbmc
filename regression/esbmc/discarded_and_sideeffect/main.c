/* The && arm: discarded `A && f()` is `if (A) f();`, so f runs iff A holds. */
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
