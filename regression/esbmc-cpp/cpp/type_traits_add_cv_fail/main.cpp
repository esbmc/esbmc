#include <type_traits>
#include <cassert>

int main()
{
  std::add_cv<int>::type x = 5;
  std::add_volatile<int>::type y = 7;
  assert(x + y == 13);
  return 0;
}
