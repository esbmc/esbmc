void check(int c)
{
  __ESBMC_assert(c, "j >= 1");
}
int main()
{
  int n = nondet_int();
  if (n > 100000)
    return 0;
  for (int j = 0; j < n; j++)
    check(j >= 1);
}
