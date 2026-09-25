int nondet_int();

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 0 && x < 10);
  int y = x + 1;
  __ESBMC_assert(y > 100, "y must exceed one hundred");
  int from = 0;
  return 0;
}
