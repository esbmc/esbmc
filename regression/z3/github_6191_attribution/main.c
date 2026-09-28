int main()
{
  int a[12], b[12], i, j;
  __ESBMC_assume(__ESBMC_forall(&j, !(0 <= j && j < 12) || b[j] == 7));
  __ESBMC_assert(__ESBMC_forall(&j, !(0 <= j && j < 12) || b[j] == 7), "holds");
  __ESBMC_assert(
    __ESBMC_forall(&i, !(0 <= i && i < 12) || a[i] != a[0]), "genuine");
  return 0;
}
