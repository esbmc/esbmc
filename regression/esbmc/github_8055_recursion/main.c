// #8055
int nondet_int();

void f(int n, int d)
{
  if (d > 0)
  {
    f(0, d - 1);
    int i = 0;
    while (i < n)
    {
      i++;
      if (i > 3)
        break;
    }
    __ESBMC_assert(i == 0, "loop not entered");
  }
  n = 0;
}

int main()
{
  f(nondet_int(), 1);
}
