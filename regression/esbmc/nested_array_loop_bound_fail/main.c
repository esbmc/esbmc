int main(void)
{
  int a[2][2];
  a[1][1] = 4;

  int s = 0;
  for (int i = 0; i < a[1][1]; i++)
    s++;

  __ESBMC_assert(s == 5, "the loop runs one time too many");
  return 0;
}
