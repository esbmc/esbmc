int nondet_int();

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x < 4);
  int i;
  for (i = 0; i < 4; i++)
  {
    __ESBMC_assert(x + i >= 0, "no_underflow");
  }
  return 0;
}
