// #2175: --function on an ESBMC goto binary must reach the named function,
// with its parameters nondet rather than bound to main's argc/argv.
int g = 7;

int foo(int x)
{
  __ESBMC_assert(g == 7, "static initialiser ran");
  if (x > 0)
    __ESBMC_assert(x >= 1, "x is positive");
  return x;
}

int main(int argc, char **argv)
{
  return foo(argc);
}
