// C11 _Atomic type: assertion should fail after modification
#include <assert.h>

int main()
{
  _Atomic int x = 0;
  x = 10;
  int val = x;
  assert(val == 0);
  return 0;
}
