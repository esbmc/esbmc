// Issue #11169: tuple_size / tuple_element for cv-qualified types.
#include <array>
#include <cassert>
#include <tuple>
#include <utility>

struct S
{
  int a;
  int b;
};

static_assert(std::tuple_size<const std::array<int, 2>>::value == 2, "");
static_assert(std::tuple_size<const std::pair<int, int>>::value == 2, "");

int main()
{
  std::array<int, 2> a{1, 2};
  const auto [p, q] = a;
  assert(p + q == 3);

  const std::pair<int, int> pr{3, 4};
  const auto [m, n] = pr;
  assert(m + n == 7);

  const std::tuple<int, int, int> t{1, 2, 3};
  const auto [x, y, z] = t;
  assert(x + y + z == 6);

  const S s{5, 6};
  const auto [u, v] = s;
  assert(u + v == 11);
  return 0;
}
