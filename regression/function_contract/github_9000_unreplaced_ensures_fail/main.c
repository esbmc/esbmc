/* #9000: an ensures on __ESBMC_return_value in a function that is neither
 * replaced nor enforced must not prune the paths through it. */
int nondet_int();
int g;
void f(int x)
{
  __ESBMC_ensures(g == x);
  __ESBMC_assigns(g);
  g = x;
}
int len(int c)
{
  __ESBMC_ensures(__ESBMC_return_value == (c <= 0xFFFF ? 1 : 2));
  __ESBMC_assigns();
  return c <= 0xFFFF ? 1 : 2;
}
int main()
{
  int x = nondet_int();
  f(x);
  int r = len(x);
  __ESBMC_assert(!(g == 5), "g==5 reachable");
  return 0;
}
