#include <tuple>
#include <cassert>

int main()
{
  int a, b;
  std::tie(a, b) = std::make_tuple(5, 10);
  assert(a == 10 && b == 5);
  return 0;
}
