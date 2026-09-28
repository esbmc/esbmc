/* #7525: the parser runs with boost's allow_guessing, so an abbreviated
 * option is still an option. `--show-loo` reaches --show-loops and is
 * swallowed as --smtlib-solver-prog's value; an exact-match check would let
 * this spelling through unwarned. */
int main(void)
{
  int x = 0;
  return x;
}
