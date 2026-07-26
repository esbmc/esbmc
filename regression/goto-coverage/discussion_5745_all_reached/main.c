int nondet_int();

int main()
{
  int a = nondet_int();
  __ESBMC_assert(a == a, "a equals itself");
  if (a > 0)
    __ESBMC_assert(a > 0, "positive branch");
  else
    __ESBMC_assert(a <= 0, "non-positive branch");
}
