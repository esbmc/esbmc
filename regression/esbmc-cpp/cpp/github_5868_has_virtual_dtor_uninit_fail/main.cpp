#include <type_traits>
#include <cassert>

struct plain
{
  int a;
};

int main()
{
  // plain has no virtual destructor.
  assert(std::has_virtual_destructor<plain>::value);
  return 0;
}
