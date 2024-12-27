#include <tuple>
#include <cassert>

int main()
{
  auto my_tuple = std::make_tuple(10, 3.14, 't');

  assert(std::get<0>(my_tuple) == 10);
  assert(std::get<1>(my_tuple) == 3.14);
  assert(std::get<2>(my_tuple) == 't');

  return 0;
}
