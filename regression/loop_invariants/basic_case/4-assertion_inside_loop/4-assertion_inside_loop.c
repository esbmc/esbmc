// this version is ignoring the integer overflow
// other tool also handle without integer overflow.
// need disscussion
int main()
{
  unsigned int x, y;
  x = 0U;
  y = 4U;
  __ESBMC_loop_invariant(y % 4 == 0 && x % 4 == 0);
  while (1)
  {
    __ESBMC_assume(x < 1000000 && y < 1000000);
    x = x + y;
    y = y + 4U;
    __ESBMC_assume(x < 1000000 && y < 1000000);

    assert(x != 3);
  }

  return 0;
}