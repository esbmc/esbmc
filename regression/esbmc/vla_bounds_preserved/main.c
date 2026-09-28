/* Anti-vacuity twin of ptr_rel_huge_object_vla: R40 bounds a symbolic VLA size
   by assumption, and an assumption that prunes more than the sizes above
   PTRDIFF_MAX would prove this reachable assert(0). */
#include <assert.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();
  if (n == 0)
    return 0;

  char a[n];
  a[0] = 1;

  assert(0);
  return 0;
}
