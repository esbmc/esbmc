#include <cstdlib>
#include <cstring>
#include <cassert>

struct has_bitfield {
  unsigned int a;
  unsigned int b : 2;
  unsigned int c : 2;
  unsigned int d : 2;
} beans;

int main(void)
{
  assert(beans.a == 0);
  assert(beans.b == 0);
  assert(beans.c == 0);
  assert(beans.d == 0);
  beans.d = 1;
  assert(beans.d == 1);
  beans.d = 0xFFFFFF;
  assert(beans.d == 3);

  memset(&beans, 0, sizeof(beans));
  assert(beans.d != 0); // should be 0
  return 0;
}
