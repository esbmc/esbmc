#include <tuple>
#include <array>
#include <cassert>

int main()
{
  std::tuple<int, char, int> t{1, 'A', 3};
  auto [a, b, c] = t;
  assert(a == 1 && b == 'A' && c == 3);

  auto [x, y] = std::make_tuple(5, 10);
  assert(x + y == 15);

  auto u = std::make_tuple(7, 8);
  auto &[p, q] = u;
  p = 99;
  assert(std::get<0>(u) == 99);

  const std::tuple<int, int> ct{4, 6};
  auto [m, n] = ct;
  assert(m + n == 10);

  std::array<int, 3> arr = {2, 4, 6};
  auto [i, j, k] = arr;
  assert(i == 2 && j == 4 && k == 6);

  return 0;
}
