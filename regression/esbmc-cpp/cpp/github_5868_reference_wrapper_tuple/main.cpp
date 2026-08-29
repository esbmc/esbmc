#include <tuple>
#include <functional>
#include <cassert>
int main()
{
  int n = 1;
  auto t = std::make_tuple(std::ref(n), 2);
  std::get<0>(t) = 5;
  assert(n == 5);
  assert(std::get<1>(t) == 2);
  return 0;
}
