/* An access wider than the object it starts in is out of bounds at every
   offset, including 0. That arm is a constant in the bounds check, so it needs
   a test of its own: the offset comparison can never reach it. Widths are
   fixed so the access stays wider than the object on LLP64 targets too. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  int32_t x = 7;
  int64_t v = *(int64_t *)((char *)&x + 0);
  assert(v == 7);
  return 0;
}
