#include <cassert>

int foo(char, int, float);

using first = __type_pack_element<0, char, int, float>;
using second = __type_pack_element<1, char, int, float>;
using third = __type_pack_element<2, char, int, float>;

int main()
{
  assert(foo(1, 2, 3) == 6);
}
int foo(first a, second b, third c)
{
  return a * b * c;
}