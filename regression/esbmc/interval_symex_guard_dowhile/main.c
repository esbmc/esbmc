/* do-while: backward GOTO guard is the loop condition.
   The interval domain assumes the condition on each back-edge,
   tightening x's upper bound until the guard is proven false. */
int main()
{
  int x;
  __ESBMC_assume(x >= 0 && x <= 2);
  do
  {
    x++;
  } while (x < 3);
  __ESBMC_assert(x >= 3, "x must be at least 3 after loop");
  return 0;
}
