#include <cmath>
#include <cassert>

int main()
{
  std::float_t v = 0.5f;
  std::double_t w = 0.5;
  assert(std::round(v) == 1.0);
  assert(std::round(w) == 1.0);
  return 0;
}
