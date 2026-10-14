/* An access wider than the object it starts in is out of bounds at every
   offset, including 0. That arm is a constant in the bounds check, so it needs
   a test of its own: the offset comparison can never reach it. */
#include <assert.h>

int main(void)
{
  int x = 7;
  long long v = *(long long *)((char *)&x + 0);
  assert(v == 7);
  return 0;
}
