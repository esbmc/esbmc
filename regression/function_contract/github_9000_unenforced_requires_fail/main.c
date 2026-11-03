/* #9000: a requires in a function that is neither replaced nor enforced
 * must not be assumed at its call sites. */
int g;
void f(int x)
{
  __ESBMC_ensures(g == x);
  __ESBMC_assigns(g);
  g = x;
}
int half(int c)
{
  __ESBMC_requires(c > 0);
  __ESBMC_ensures(1);
  __ESBMC_assert(c > 0, "half needs a positive argument");
  return c / 2;
}
int main()
{
  f(0);
  half(g);
  return 0;
}
