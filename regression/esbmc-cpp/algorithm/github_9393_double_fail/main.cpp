// #9393: std::max(double, double) returns the first argument on a tie
#include <algorithm>
#include <cassert>
#include <cmath>

int main()
{
  assert(std::signbit(std::max(0.0, -0.0)));
  return 0;
}
