int main()
{
  int a[4];
  for (int i = 0; i < 3; i++)
    a[i] = 0;
  __ESBMC_assert(a[3] == 0, "a[3]");
}
