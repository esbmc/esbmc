#include <stdio.h>
#include <assert.h>

double acceleration(double fx, double fy)
{
  return fx / 1.0, fy / 2.0, fx / fy;
}

int main()
{
  double t = 0;
  double fx = 4.0;
  double fy = 4.0;

  double ax = acceleration(fx, fy);
  assert(ax == 1.0);
  t = fx++, fx + 1, fx + 2;
  assert(t == 4.0);
  assert(fx == 5.0);

  return 0;
}