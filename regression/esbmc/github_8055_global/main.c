// #8055
int nondet_int();
int g;

int f(void)
{
  g = 0;
  return nondet_int();
}

int main()
{
  int i = 0;
  g = f();
  while (i < g)
  {
    i++;
    if (i > 3)
      break;
  }
  __ESBMC_assert(i == 0, "loop not entered");
}
