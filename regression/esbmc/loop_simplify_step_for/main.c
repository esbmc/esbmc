/* Pins step recognition for the ascending counter pattern:
 * for(i=0; i<17; i++) ; should rewrite to i = 17. The strong
 * assertion `i == 17` only holds if the rewrite produces the exact
 * post-value — a havoc + assume(!(i<17)) rewrite would only give
 * us i >= 17 and the assertion would fail. Symex then sees one
 * ASSIGN, no loop, no unwinding. */
int main()
{
  int i;
  for (i = 0; i < 17; i++)
    ;
  __ESBMC_assert(i == 17, "step recognition pins exact post-value");
  return 0;
}
