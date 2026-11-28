// Issue #11169: a wrong value bound through a const structured binding is
// still reported.
#include <array>
#include <cassert>

int main()
{
  std::array<int, 2> a{1, 2};
  const auto [p, q] = a;
  assert(p + q == 4);
  return 0;
}
