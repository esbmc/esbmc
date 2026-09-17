#include <type_traits>
#include <cassert>

int main()
{
  // int[3][4] has two dimensions, not one.
  assert(std::rank<int[3][4]>::value == 1);
  return 0;
}
