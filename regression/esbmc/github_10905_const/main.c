/* #10905: a pointer made from an integer equals a pointer at its address */
#include <assert.h>
#include <stdint.h>
int main(void)
{
  uint32_t buf[1];
  uint32_t *p1 = buf;
  uint32_t *p2 = (uint32_t *)(uintptr_t)4 + 3;
  if ((intptr_t)p1 == (intptr_t)p2)
    assert(!(p1 != p2));
  return 0;
}
