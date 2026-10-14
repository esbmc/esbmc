#include <stdint.h>

/* Once `&sp.b` is cast to `uint64_t *` the packed provenance that justifies
 * skipping the alignment check is gone, and the base of a packed object is
 * unconstrained -- so the load really can be misaligned; see #7707. */

struct __attribute__((packed)) S
{
  char a;
  char pad[7];
  uint64_t b;
};

int main(void)
{
  struct S sp;
  uint64_t *p = (uint64_t *)&sp.b;
  uint64_t z = *p;
  (void)z;
  return 0;
}
