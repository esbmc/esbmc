
int main()
{
  unsigned int x, y;
  x = 0U;
  y = 4U;
  //__ESBMC_loop_invariant(x % 4 == 0);
  __ESBMC_loop_invariant(y % 4 == 0);
  while (1)
  {
    x = x + y;
    y = y + 4U;

    assert(x != 30);
  }

  return 0;
}