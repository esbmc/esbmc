int main()
{
  int x[1][1][3];
  x[0][0][0] = 80;
  x[0][0][2] = 0;
  __ESBMC_assume((x[0][0][0] >= 77) && (x[0][0][0] <= 87));
  __ESBMC_assume((x[0][0][2] >= -64) && (x[0][0][2] <= 64));
  __ESBMC_assert(1, "");
  return 0;
}