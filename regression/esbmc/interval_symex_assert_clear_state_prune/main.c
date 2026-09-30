// After a write through a pointer resets the interval domain, a later assume
// is tracked again and discharges the claim.
int nondet_int();
int main()
{
  int y, z;
  int *p = &y;
  z = 1;
  *p = nondet_int();
  __ESBMC_assume(y >= 0 && y <= 5);
  __ESBMC_assert(y <= 10, "y at most 10");
}
