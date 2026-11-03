// #8743: k == Y must be reachable for an enum with a narrow unsigned
// underlying type
#include <cassert>
unsigned nondet_uint();
enum K : unsigned char { Y = 200 };
int main()
{
  K k = static_cast<K>(nondet_uint());
  if (k == Y)
    assert(false);
  return 0;
}
