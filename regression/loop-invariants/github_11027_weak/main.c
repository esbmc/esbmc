/* Regression: GitHub #11027 -- x == 10 holds after the loop, but the
 * invariant does not pin x, and no run from the loop's entry state leaves the
 * loop at once: the claim stays unknown. Its trace still ends at the property
 * it violates. */
int main()
{
  int x = 0;
  int i;
  __ESBMC_loop_invariant(i >= 0 && i <= 10 && x >= 0 && x <= i);
  for (i = 0; i < 10; ++i)
    x++;
  __ESBMC_assert(x == 10, "after: x == 10");
}
