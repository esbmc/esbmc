// #8743: a branch on an enumerator, explicit or implicit, above the signed
// range of the enum's unsigned char underlying type is reachable.
#include <cassert>
unsigned nondet_uint();
enum K : unsigned char { Y = 200, Y1 };
int main()
{
  K k = static_cast<K>(nondet_uint());
  if (k == Y)
    assert(false);
  if (k == Y1)
    assert(false);
  return 0;
}
