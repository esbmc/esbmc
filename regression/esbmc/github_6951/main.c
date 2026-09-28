#include <stdint.h>

/* The base-address constraint must cover the access widths check_alignment()
 * assumes; see #6951. */

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
