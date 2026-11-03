// github #9368: posix_memalign(&p, 16, 24) succeeds, so the write one past
// the 24-byte object is reachable.
#include <stdlib.h>

int main(void)
{
  char *p;
  if (posix_memalign((void **)&p, 16, 24) != 0)
    return 0;
  p[24] = 1;
  free(p);
  return 0;
}
