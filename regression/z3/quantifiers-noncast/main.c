int main()
{
  char idx;
  int prop = __ESBMC_forall(&idx, idx != idx + 1);
  __ESBMC_assert(prop, "prop");
  return 0;
}
