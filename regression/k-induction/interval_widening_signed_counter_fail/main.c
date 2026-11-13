// The bound must not hide a real violation.
int a[100];
int main()
{
  a[3] = 1;
  for (int j = 0; j < 100; j++)
    __ESBMC_assert(a[j] == 0, "zero");
}
