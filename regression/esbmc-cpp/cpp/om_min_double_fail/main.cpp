// A truncating std::min(double, double) makes this wrong assertion hold.
#include <algorithm>
#include <cassert>

int main()
{
  assert(std::min(0.5, 2.0) == 0);
  return 0;
}
