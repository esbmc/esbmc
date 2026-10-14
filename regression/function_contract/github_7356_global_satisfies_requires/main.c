/* The non-vacuous half of #7356. `requires` is satisfiable at the initialiser,
 * so one VCC is generated and the vacuity probe finds nothing, yet the state
 * that refutes `ensures` is g == 6, which enforcement never reached while the
 * globals kept their static initialisers.
 */
int g;

void f(void)
{
  __ESBMC_requires(g >= 0);
  __ESBMC_assigns(g);
  __ESBMC_ensures(g != 7);
  if (g == 6)
    g = 7;
}

int main(void)
{
  return 0;
}
