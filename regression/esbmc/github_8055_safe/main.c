// #8055
int nondet_int();

int f(void)
{
  int r = nondet_int();
  __ESBMC_assume(r <= 0);
  return r;
}

void g(int n)
{
  int i = 0;
  while (i < n)
  {
    i++;
    if (i > 3)
      break;
  }
  __ESBMC_assert(i == 0, "loop not entered");
  n = 5;
}

int main()
{
  int n = 0, i = 0;
  n = f();
  while (i < n)
  {
    i++;
    if (i > 3)
      break;
  }
  __ESBMC_assert(i == 0, "loop not entered");
  g(0);
  g(-nondet_int() * 0);
}
