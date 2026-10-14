// #8047: after joining both paths, x is still in [0, 10], so the claim is
// discharged by the interval domain.
int main()
{
  int x;
  __ESBMC_assume(x >= 0 && x <= 10);
  if (x == 5)
    ;
  else
    __ESBMC_assume(x == 7);
  __ESBMC_assert(x <= 10, "x at most 10");
}
