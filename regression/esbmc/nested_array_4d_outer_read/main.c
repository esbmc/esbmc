int nondet_int(void);

int main(void)
{
  int a[2][2][2][2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++)
        for (int w = 0; w < 2; w++)
          a[x][y][z][w] = 1;

  a[1][1][1][1] = 4;

  __ESBMC_assert(a[i][1][1][1] >= 1, "four dimensions read back");
  return 0;
}
