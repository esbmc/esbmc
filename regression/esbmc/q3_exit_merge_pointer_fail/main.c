int nondet_int();

int main()
{
  int a = 1, b = 2, n = 0;
  int *p = &a;
  while (nondet_int())
  {
    n++;
    if (nondet_int())
    {
      p = &b;
      break;
    }
  }
  __ESBMC_assert(*p == 1, "p points to a or b");
  __ESBMC_assert(n >= 0, "n is not negative");
  return 0;
}
