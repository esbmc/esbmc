// #8055
int nondet_int();

void f(int n)
{
  int i = 0;
  while (i < n)
  {
    i++;
    if (i > 3)
      break;
  }
  __ESBMC_assert(i == 0, "loop not entered");
  n = 0;
}

int main()
{
  f(0);
  f(nondet_int());
}
