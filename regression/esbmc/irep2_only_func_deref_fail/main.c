/* The re-taken address designates the function it came from, so claiming a
   different one must be caught. */
int f(void)
{
  return 7;
}

int g(void)
{
  return 8;
}

int main(void)
{
  /* Repeated, so the arm has to fire more than once. */
  __ESBMC_assert(***f == &g, "*f is f, not g");
  return 0;
}
