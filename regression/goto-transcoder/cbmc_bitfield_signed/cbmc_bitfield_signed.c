#include <assert.h>
// A signed 3-bit bitfield: storing 5 wraps to -3 (two's complement in 3 bits).
struct S
{
  int a : 3;
};
int main()
{
  struct S s;
  s.a = 5;
  assert(s.a == -3);
  return 0;
}