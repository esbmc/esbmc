// Pin issue #4475: <utility> must parse and verify cleanly under --std c++03.
// Exercises the C++03 fallback pair/make_pair/swap and the relational
// operators routed through the OM_CONSTEXPR macro.
#include <cassert>
#include <utility>

int main()
{
  std::pair<int, int> p1(1, 2);
  std::pair<int, int> p2 = std::make_pair(1, 2);
  std::pair<int, int> p3(3, 4);

  assert(p1.first == 1 && p1.second == 2);
  assert(p1 == p2);
  assert(p1 != p3);
  assert(p1 < p3);
  assert(p3 > p1);
  assert(p1 <= p2);
  assert(p2 >= p1);

  int a = 5, b = 10;
  std::swap(a, b);
  assert(a == 10 && b == 5);

  p1.swap(p3);
  assert(p1.first == 3 && p3.first == 1);

  return 0;
}
