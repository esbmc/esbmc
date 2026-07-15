#include <assert.h>
// A bitfield member holds only its declared width; assigning 9 to a 3-bit
// unsigned field keeps the low 3 bits (9 & 7 == 1).
struct S
{
  unsigned a : 3;
  unsigned b : 5;
};
int main()
{
  struct S s;
  s.a = 9;
  s.b = 20;
  assert(s.a == 1 && s.b == 20);
  return 0;
}