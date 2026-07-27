int main()
{
  int a[12];
  int i;
  __ESBMC_assert(
    __ESBMC_forall(&i, !(0 <= i && i < 12) || a[i] != a[0]), "false forall");
  return 0;
}
