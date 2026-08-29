int main()
{
  int i, j;

  i = 3;
  j = 7;

  __ESBMC_assert(
    __ESBMC_forall(&i, !(0 <= i && i < 4) || __ESBMC_exists(&j, j == i)),
    "nested exists eq");
}
