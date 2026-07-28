// <array> is a C++11 header, but its relational operators were unconditionally
// constexpr while looping, which a C++11 constexpr body may not do, so the
// whole header failed to parse under --std c++11 (github #3387).
#include <array>
#include <cassert>

int main()
{
  std::array<int, 3> a = {{1, 2, 3}};
  std::array<int, 3> b = {{1, 2, 3}};
  std::array<int, 3> c = {{1, 2, 4}};

  assert(a == b);
  assert(a != c);
  assert(a < c);
  assert(c > a);
  assert(a <= b);
  assert(a >= b);

  assert(a.size() == 3);
  assert(a.at(1) == 2);
  a.fill(7);
  assert(a[0] == 7 && a[2] == 7);

  return 0;
}
