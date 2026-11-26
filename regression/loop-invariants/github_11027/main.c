/* Regression: GitHub #11027 -- x is not written in the loop, so the first
 * iteration from the entry state violates the assertion: a real failure, not
 * an invariant too weak to prove it, and its trace names the property. */
int main()
{
  int x = 0;
  int idx;
  __ESBMC_loop_invariant(x == 0);
  for (idx = 0; idx < 10; ++idx)
  {
    __ESBMC_assert(x == 1, "top: x==1");
  }
}
