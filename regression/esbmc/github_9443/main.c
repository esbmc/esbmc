#include <assert.h>

struct S
{
  unsigned x : 3;
  signed y : 4;
} s;

int main(void)
{
  s.x = nondet_uint();
  unsigned old = s.x;
  unsigned r = s.x++;
  assert(r == old && r < 8);
  assert(s.x == ((old + 1) & 7));

  s.y = nondet_int();
  int y = s.y;
  long long q = s.y--;
  assert(q == y && q >= -8 && q < 8);
  return 0;
}
