#include <assert.h>
// A _Bool:1 bitfield is unsigned 1-bit: value 1 reads back as 1, not -1.
struct S
{
  _Bool a : 1;
};
int main()
{
  struct S s;
  s.a = 1;
  assert(s.a == 1);
  return 0;
}