// #8055
int nondet_int();

int f(void)
{
  int r = nondet_int();
  return r;
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
}
