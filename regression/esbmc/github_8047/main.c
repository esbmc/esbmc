// #8047: the else-branch assume narrowed x in the shared interval domain, and
// the merge did not join the other path back, so x > -7 was pruned as TRUE.
int main()
{
  int x;
  if (x == -128)
    ;
  else
    __ESBMC_assume(x && x == 7);
  __ESBMC_assert(x > -7, "x above -7");
}
