int nondet_int();

int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a == -7);
  __ESBMC_assert(a / 2 == -4 || a % 2 == 1, "Euclidean rounding");
  return 0;
}
