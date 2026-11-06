int nondet_int();
int g;

int f(int a, int b)
{
  if (a)
    return 1;
  if (b)
    g++;
  return 0;
}

int main()
{
  f(nondet_int(), nondet_int());
  f(nondet_int(), nondet_int());
  __ESBMC_assert(g == 0, "g");
  return 0;
}
