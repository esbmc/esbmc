#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

/* #9368: POSIX constrains only the alignment of posix_memalign, and C17
 * aligned_alloc no longer requires size to be a multiple of it. */
int main(void)
{
  void *p = 0, *q = 0;
  int r = posix_memalign(&p, 16, 24);
  assert(r == 0 || r == ENOMEM);
  assert(r != 0 || ((uintptr_t)p & 15) == 0);
  int s = posix_memalign(&q, 4, 8);
  assert(s == EINVAL);
  char *a = aligned_alloc(16, 24);
  assert(a);
  a[23] = 1;
  free(a);
  free(p);
  return 0;
}
