/* A signed loop that may never be entered. Without the "entered or still at
 * i0" conjunct the exit admits i == n with n negative, and the closed form then
 * reports s == n < 0 -- a state the loop cannot produce, failing the user's own
 * assertion after the loop. Note the assertion is deliberately one the spurious
 * state violates: an assertion like `s == n || s == 0` would absorb it and pass
 * for the wrong reason. */
int nondet_int();
int main(void)
{
  int n = nondet_int();
  int i = 0, s = 0;
  while (i < n)
  {
    s = s + 1;
    i++;
  }
  __ESBMC_assert(s >= 0, "a count is never negative");
  return 0;
}
