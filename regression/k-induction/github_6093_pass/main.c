int main()
{
  int i, s = 0;
  for (i = 0; i < 3; i++)
    s++;
  __ESBMC_assert(s == 3, "s_eq_3");
}
