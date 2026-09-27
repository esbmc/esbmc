/* The counterpart of github_7356_global_satisfies_requires. The ensures leans
 * on the requires: without `g >= 0` the entry state g == -1 leaves g negative
 * on exit. So it holds for every state requires admits, and enforcement must
 * still prove it once the globals are havocked.
 */
int g;

void f(void)
{
  __ESBMC_requires(g >= 0);
  __ESBMC_assigns(g);
  __ESBMC_ensures(g >= 0 && g != 6);
  if (g == 6)
    g = 7;
}

int main(void)
{
  return 0;
}
