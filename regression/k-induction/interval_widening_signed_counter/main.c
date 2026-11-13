// Widening at a guard inside the body dropped j >= 0, so the inductive step
// started from j = -1 and read a[-1].
int a[100];
int main()
{
  for (int j = 0; j < 100; j++)
    __ESBMC_assert(a[j] == 0, "zero");
}
