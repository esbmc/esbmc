void check(int c)
{
  __ESBMC_assert(c, "i >= 1");
}
void run(int i, int n)
{
  while (i < n)
  {
    check(i >= 1);
    i++;
  }
}
int main()
{
  int n = nondet_int();
  if (n > 100000)
    return 0;
  run(0, n);
}
