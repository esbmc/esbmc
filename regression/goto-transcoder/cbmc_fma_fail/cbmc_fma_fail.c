extern double fma(double, double, double);
int main()
{
  double z = fma(2.5, 3.0, 1.0);
  __CPROVER_assert(z == 9.0, "fma(2.5,3.0,1.0) != 9.0");
  return 0;
}
