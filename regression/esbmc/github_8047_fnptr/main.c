// #8047: the assume in h narrows g in the shared interval domain; the merge
// after the function-pointer call must join the path through f back in.
int g;
int nondet_int();

int f(int y)
{
  return y;
}

int h(int y)
{
  __ESBMC_assume(g == 7);
  return y;
}

int main()
{
  int c;
  g = nondet_int();
  int (*fp)(int) = c ? f : h;
  fp(0);
  __ESBMC_assert(g > -7, "g above -7");
}
