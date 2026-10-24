// #8047: every path parked at a function-pointer call records its domain, so
// x stays in [0, 10] across the merge and the claim is discharged by it.
int f(int y)
{
  return y + 1;
}

int g(int y)
{
  return y;
}

int main()
{
  int x, c;
  __ESBMC_assume(x >= 0 && x <= 10);
  int (*fp)(int) = c ? f : g;
  fp(x);
  __ESBMC_assert(x <= 10, "x at most 10");
}
