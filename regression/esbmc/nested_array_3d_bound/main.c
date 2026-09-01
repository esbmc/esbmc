int main(void)
{
  int a[2][2][2];
  a[1][1][0] = 5;
  a[1][1][1] = 4;

  int s = 0;
  for (int i = 0; i < a[1][1][1]; i++)
    s++;

  __ESBMC_assert(s == 4, "a bound stored in a 3-D element decides the loop");
  __ESBMC_assert(a[1][1][0] == 5, "the row's earlier store survives");
  return 0;
}
