/* The result is the only definition of what the call returns, so it is a fresh
 * nondet. For a pointer that leaves it without a value set: `p == &a` is proved
 * from the ensures, but the write through p does not reach a.
 *
 * Leaving the result undeclared instead is what #7009 was: the ensures would be
 * assumed about the caller's stale pointer, `p == &a` would be assume-false and
 * everything after the call would pass vacuously. A reported failure is the
 * safe direction of the two, and unlike the other it is visible. */
int a;

int *pick(int c)
{
  __ESBMC_requires(c == 0);
  __ESBMC_ensures(__ESBMC_return_value == &a);

  return &a;
}

int main(void)
{
  int *p = 0;
  p = pick(0);
  __ESBMC_assert(p == &a, "pointer equality holds");
  *p = 5;
  __ESBMC_assert(a == 5, "the write through p reaches a");
  return 0;
}
