#include <tuple>
#include <cassert>

int main()
{
  int a, b, c;
  std::tie(a, b) = std::make_tuple(5, 10);
  assert(a == 5 && b == 10);

  std::tie(a, b, c) = std::make_tuple(1, 2, 3);
  assert(a == 1 && b == 2 && c == 3);

  int only;
  std::tie(only, std::ignore) = std::make_tuple(42, 99);
  assert(only == 42);

  int x = 1, y = 2;
  std::tie(x, y) = std::make_tuple(y, x);
  assert(x == 2 && y == 1);

  std::tuple<int, int> t{7, 8};
  std::tie(a, b) = t;
  assert(a == 7 && b == 8);

  return 0;
}
