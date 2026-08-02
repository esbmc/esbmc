int nondet_int();

int main()
{
  int a = nondet_int();
  __ESBMC_assume(a % 2 == 0);
  if (a % 2)
  {
    ++a;
    __ESBMC_assert(0, "unreachable odd branch");
  }
  __ESBMC_assert(a % 2 == 0, "a is even here");
  return 0;
}
