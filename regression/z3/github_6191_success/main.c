int zeroed[12];

int main()
{
  int i;
  __ESBMC_assert(
    __ESBMC_forall(&i, !(0 <= i && i < 12) || zeroed[i] == 0),
    "true forall");
  return 0;
}
