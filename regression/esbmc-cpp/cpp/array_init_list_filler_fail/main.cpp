// The elements past an array's braced list are initialised as if from an
// empty list: by default member initializers, not zeroed.
#include <cassert>
struct S
{
  int a = 5;
  int b;
};
int main()
{
  S s[2]{S{1, 2}};
  assert(s[1].a == 0);
  return 0;
}
