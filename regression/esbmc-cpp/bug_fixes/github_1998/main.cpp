#include <cmath>
#include <cassert>

float max(float a, float b)

{
  assert(std::isnan(a) && "Max<float> A was NaN!");

  assert(std::isnan(b) && "Max<float> B was NaN!");

  return ((a > b) ? a : b);
}
int main()
{
  float n1 = NAN, n2 = NAN;
  return max(n1, n2);
}
