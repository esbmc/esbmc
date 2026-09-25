#include <stdint.h>

/* The base is aligned, so the first claim holds -- an implementation that left
 * the base unconstrained would report that one instead, and this test would
 * not reach the violation it pins. The offset within the object is still
 * checked, so the odd offset is a misaligned access either way. */

int main(void)
{
  char buf[16];
  uint16_t *base = (uint16_t *)(buf + 0);
  __ESBMC_assert(((uintptr_t)base % 2) == 0, "buffer base is 2-aligned");

  uint16_t *odd = (uint16_t *)(buf + 1);
  uint16_t x = *odd;
  (void)x;
  return 0;
}
