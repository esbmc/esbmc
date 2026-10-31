#include <stdlib.h>

/* #9368: posix_memalign(16, 24) is valid, so the write one past the 24-byte
 * object must be reported. */
int main(void)
{
  char *p;
  if (posix_memalign((void **)&p, 16, 24) != 0)
    return 0;
  p[24] = 1;
  free(p);
  return 0;
}
