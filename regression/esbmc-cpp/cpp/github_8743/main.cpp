// #8743: enumerators of an enum with a narrow unsigned underlying type
#include <cassert>
unsigned short nondet_ushort();
enum K : unsigned char { Y = 200 };
enum class S : unsigned short { Z = 40000 };
enum L : unsigned char { A = 199, B };
enum W : unsigned long long { P = 0x100000000ULL, Q };
int main()
{
  int y = Y;
  assert(y == 200);
  assert((int)S::Z == 40000);
  int b = B;
  assert(b == 200);
  assert(Q == 0x100000001ULL);
  S s = static_cast<S>(nondet_ushort());
  switch (s)
  {
  case S::Z:
    assert((unsigned)s == 40000);
    break;
  default:
    assert((unsigned)s != 40000);
  }
  return 0;
}
