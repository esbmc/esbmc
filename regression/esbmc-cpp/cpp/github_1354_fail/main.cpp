#include <cmath>
#include <cassert>

int main()
{
  std::float_t v = 0.4f;
  std::double_t w = 0.4;
  assert(std::round(v) == 1.0);
  assert(std::round(w) == 1.0);
  return 0;
}
