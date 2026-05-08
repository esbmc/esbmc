// Two independent if statements — exercises k=2 prefix.
// Branch coverage needs 2 tests; k=2 path coverage exposes 4 paths
// (TT, TF, FT, FF) — all feasible.
int main()
{
  int a, b;
  if (a > 0)
    a = 1;
  else
    a = -1;

  if (b > 0)
    b = 1;
  else
    b = -1;

  return a + b;
}
