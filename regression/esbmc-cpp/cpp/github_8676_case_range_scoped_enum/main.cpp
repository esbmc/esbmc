#include <cassert>
enum class E : unsigned char { A, B, C, D, F };
unsigned nondet_uint();
int main()
{
  E e = static_cast<E>(nondet_uint() % 5);
  int r = 0;
  switch (e)
  {
  case E::A ... E::C:
    r = 1;
    break;
  default:
    r = 2;
  }
  assert(r == 1 || e == E::D || e == E::F);
  assert(r == 2 || e == E::A || e == E::B || e == E::C);
  return 0;
}
