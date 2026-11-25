/* #10856: memcpy's loop is skipped, and the loop after it never exits. */
#include <string.h>

int main(void)
{
  char d[4];
  memcpy(d, 0, 0);
  unsigned x = 1;
  while (x != 0)
    x |= 1;
  return 0;
}
