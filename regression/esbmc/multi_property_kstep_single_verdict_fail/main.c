int nondet_int();

int main()
{
  int x = nondet_int();
  __ESBMC_assert(x != 1, "first");
  __ESBMC_assert(x != 2, "second");
  return 0;
}
