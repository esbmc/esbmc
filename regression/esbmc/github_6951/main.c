#include <stdint.h>

/* dereferencet::check_alignment() decides a scalar access from the offset
 * within the object alone, so the address-space model must give every
 * non-packed object at least the ABI's fundamental alignment; otherwise the
 * two disagree on whether the same pointer can be misaligned. */

char g_buf[16];

int main(void)
{
  char buf[16];
  uint16_t *p = (uint16_t *)(buf + 0);
  uint32_t *q = (uint32_t *)(g_buf + 0);
  uint16_t x = *p;
  uint32_t y = *q;
  (void)x;
  (void)y;

  __ESBMC_assert(((uintptr_t)p % 2) == 0, "automatic char buffer is 2-aligned");
  __ESBMC_assert(((uintptr_t)q % 4) == 0, "static char buffer is 4-aligned");
  return 0;
}
