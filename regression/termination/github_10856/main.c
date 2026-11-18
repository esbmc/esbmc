#include <string.h>

int main(void)
{
  char d[4];
  char *p = 0;
  memcpy(d, 0, 0);
  memset(p, 0, 0);
  unsigned x = 1;
  while (x != 0)
    x <<= 1;
  return 0;
}
