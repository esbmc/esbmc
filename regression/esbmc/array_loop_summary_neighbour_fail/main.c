int a[50];
int main()
{
  a[0] = 1;
  for (int i = 1; i < 50; i++)
    a[i] = a[i - 1] + 1;
  __ESBMC_assert(a[49] == 49, "a[49] == 49");
}
