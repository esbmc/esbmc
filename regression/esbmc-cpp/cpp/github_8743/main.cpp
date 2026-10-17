// #8743: an enumerator of an enum with an unsigned narrow underlying type
// keeps its unsigned value when compared with a nondet enum object.
#include <cassert>
unsigned nondet_uint();
enum K : unsigned char { Y = 200, Y1 };
enum class L : unsigned short { A = 1, Z = 40000 };
int main()
{
  K k = static_cast<K>(nondet_uint());
  if (k == Y)
    assert((int)k == 200);
  else
    assert((int)k != 200);
  if (k == Y1)
    assert((int)k == 201);
  else
    assert((int)k != 201);

  L l = static_cast<L>(nondet_uint());
  switch (l)
  {
  case L::Z:
    assert((int)l == 40000);
    break;
  default:
    assert((int)l != 40000);
    break;
  }
  return 0;
}
