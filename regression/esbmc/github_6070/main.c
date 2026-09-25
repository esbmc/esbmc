int main()
{
  int i;
  for (i = 0; i < 4; i++)
  {
    __ESBMC_assert(i != 1, "fail_when_i_is_1");
    __ESBMC_assert(i != 3, "fail_when_i_is_3");
  }
  return 0;
}
