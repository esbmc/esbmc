#include <assert.h>
// A struct defined inside a function body: CBMC scope-qualifies the type
// symbol name (main::1::tag-S), which the tag reference must resolve to.
int main()
{
  struct S
  {
    int a;
    int b;
  } s;
  s.a = 3;
  s.b = 4;
  assert(s.a + s.b == 7);
  return 0;
}