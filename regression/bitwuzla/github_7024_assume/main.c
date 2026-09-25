int main()
{
  int a[10];
  int x;

  x = 11;

  __ESBMC_assume(
    __ESBMC_forall(&x, !(0 <= x && x < 10) || a[x] == 0));

  __ESBMC_assert(a[3] == 0, "assumed forall constrains a[3]");
}
