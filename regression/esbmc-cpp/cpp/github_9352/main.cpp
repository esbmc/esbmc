// #9352: a store to a bool bit-field converts the value to bool
// ([conv.bool], [class.bit]p4) instead of keeping its low bit.
#include <cassert>

struct S
{
  bool b : 1;
} s;

int main()
{
  s.b = false;
  s.b |= 2;
  assert(s.b);

  s.b = false;
  s.b += 2;
  assert(s.b);
  return 0;
}
