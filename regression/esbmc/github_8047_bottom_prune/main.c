// #8047: the else-branch assume empties the domain; the merge must replace
// that bottom with the other path's domain, so the claim is still discharged.
int main()
{
  int x;
  __ESBMC_assume(x >= 0 && x <= 10);
  if (x == 5)
    ;
  else
    __ESBMC_assume(x == 20);
  __ESBMC_assert(x <= 10, "x at most 10");
}
