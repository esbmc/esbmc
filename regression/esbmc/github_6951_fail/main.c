#include <stdint.h>

/* The offset within the object is still checked: an odd offset into an
 * otherwise well-aligned buffer remains a misaligned access. */

int main(void)
{
  char buf[16];
  uint16_t *p = (uint16_t *)(buf + 1);
  uint16_t x = *p;
  (void)x;
  return 0;
}
