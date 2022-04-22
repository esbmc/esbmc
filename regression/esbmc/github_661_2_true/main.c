int main()
{
  int x[1][1][3];
  x[0][0][0] = nondet_int();
  x[0][0][1] = nondet_int();
  x[0][0][2] = nondet_int();
  __ESBMC_assume((x[0][0][0] >= 77) && (x[0][0][0] <= 87));
  __ESBMC_assume((x[0][0][1] >= -64) && (x[0][0][1] <= 64));
  __ESBMC_assume((x[0][0][2] >= -64) && (x[0][0][2] <= 64));
  __ESBMC_assert(1, "");
  return 0;
}