#include <assert.h>

struct S
{
  unsigned x : 3;
} s;

int main(void)
{
  s.x = nondet_uint();
  unsigned r = s.x++;
  assert(r != 7);
  return 0;
}
