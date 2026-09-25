int main()
{
  int idx_vfy;

  idx_vfy = 17;

  __ESBMC_assert(
    __ESBMC_forall(&idx_vfy, !(0 <= idx_vfy && idx_vfy < 10) || idx_vfy == 13),
    "forall eq");
}
