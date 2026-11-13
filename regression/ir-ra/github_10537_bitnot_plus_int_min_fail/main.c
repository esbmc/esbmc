int nondet_int(void);
int main()
{
  /* ~b + C becomes (C - 1) - b; C - 1 must not wrap at C = INT_MIN. */
  int b = nondet_int();
  __ESBMC_assume(b >= 0 && b <= 10);
  int y = ~b + (-2147483647 - 1);
  __ESBMC_assert(y > 0, "y > 0");
  return 0;
}
