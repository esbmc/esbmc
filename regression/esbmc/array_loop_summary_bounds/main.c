// The write past the end is skipped, so the summary must not check it.
int a[50];
int main()
{
  for (int i = 0; i < 60; i++)
    if (i < 50)
      a[i] = 1;
  __ESBMC_assert(a[49] == 1, "last element");
}
