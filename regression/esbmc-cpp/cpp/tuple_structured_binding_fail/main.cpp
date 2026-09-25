#include <tuple>
#include <cassert>

int main()
{
  std::tuple<int, int> t{5, 10};
  auto [a, b] = t;
  assert(a == 10 && b == 5); // wrong order on purpose
  return 0;
}
