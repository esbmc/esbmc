extern double nearbyint(double);
int main()
{
  double y = nearbyint(2.5);
  __CPROVER_assert(y == 3.0, "nearbyint(2.5) != 3.0");
  return 0;
}
