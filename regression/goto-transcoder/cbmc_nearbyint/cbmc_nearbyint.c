extern double nearbyint(double);
int main()
{
  double y = nearbyint(2.5); // round-half-to-even -> 2.0
  __CPROVER_assert(y == 2.0, "nearbyint(2.5) == 2.0");
  return 0;
}
