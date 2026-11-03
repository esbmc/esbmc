// github #9368: posix_memalign accepts any size and rejects an alignment
// that is not a multiple of sizeof(void *).
#include <assert.h>
#include <errno.h>
#include <stdlib.h>

int main(void)
{
  void *p = 0, *q = 0;
  int r = posix_memalign(&p, 16, 24);
  assert(r == 0 || r == ENOMEM);
  int s = posix_memalign(&q, 4, 8);
  assert(s == EINVAL);
  free(p);
  return 0;
}
