/* #9000: dropping the clauses of a function that is neither replaced nor
 * enforced keeps a body assertion that directly precedes one of them. */
int g;
void f(int x)
{
  __ESBMC_ensures(g == x);
  __ESBMC_assigns(g);
  g = x;
}
int half(int c)
{
  __ESBMC_assert(c > 0, "half needs a positive argument");
  __ESBMC_ensures(__ESBMC_return_value == c / 2);
  return c / 2;
}
int main()
{
  f(0);
  half(g);
  return 0;
}
