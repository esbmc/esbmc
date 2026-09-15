int main(void)
{
  int a[2][2];
  a[1][1] = 4;

  int s = 0;
  for (int i = 0; i < a[1][1]; i++)
    s++;

  __ESBMC_assert(s == 4, "a bound stored in a 2-D element decides the loop");
  return 0;
}
