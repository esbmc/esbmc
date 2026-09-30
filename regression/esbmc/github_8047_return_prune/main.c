// #8047: the domain recorded at the early return keeps x in [0, 10] across the
// end-of-function merge, so the claim is discharged by the interval domain.
int f(int x)
{
  if (x == 5)
    return x;
  __ESBMC_assume(x == 7);
  return x;
}

int main()
{
  int x;
  __ESBMC_assume(x >= 0 && x <= 10);
  f(x);
  __ESBMC_assert(x <= 10, "x at most 10");
}
