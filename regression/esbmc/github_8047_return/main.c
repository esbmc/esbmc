// #8047: an early return parks a path at the end of the function; its domain
// must be joined back there too.
int g;
int nondet_int();

void f()
{
  if (g == -128)
    return;
  __ESBMC_assume(g == 7);
}

int main()
{
  g = nondet_int();
  f();
  __ESBMC_assert(g > -7, "g above -7");
}
