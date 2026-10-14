// Neighbours of the #626 identities that do not hold. If a fold ever rewrites
// x + x to the wrong shift, or drops the -1 from ~B == -B - 1, one of these
// stops being refutable.
int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  int b = nondet_int();
  __ESBMC_assert(x + x == (x << 2), "x + x is not x << 2");
  __ESBMC_assert(~b + 1 == b, "~b + 1 is not b");
  return 0;
}
