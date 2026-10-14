extern double fma(double, double, double);
int main()
{
  double z = fma(2.5, 3.0, 1.0); // 2.5*3.0 + 1.0 = 8.5
  __CPROVER_assert(z == 8.5, "fma(2.5,3.0,1.0) == 8.5");
  return 0;
}
