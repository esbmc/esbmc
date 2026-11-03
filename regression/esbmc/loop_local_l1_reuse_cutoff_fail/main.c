int nondet_int();
int g(int d)
{
  if (d == 3)
    return nondet_int();
  int a = 0, b = 0;
  for (int i = 0; i < 2; i++)
  {
    int r = g(d + 1);
    if (i == 0)
      a = r;
    else
      b = r;
  }
  if (d == 2)
    __ESBMC_assert(a == b, "two calls may return different values");
  return a;
}
int main()
{
  return g(2);
}
