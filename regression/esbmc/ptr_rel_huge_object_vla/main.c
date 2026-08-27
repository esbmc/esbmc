/* R40: a VLA declaration is bounded only against the size computation
   overflowing the address space (goto_convert.cpp's "VLA array size in bytes
   overflows address space size"), never against PTRDIFF_MAX, so an object above
   it puts one-past-the-end in the below-base window of the pointer comparator --
   R37's witness through a third allocation path. */
#include <assert.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();
  if (n == 0)
    return 0;

  char a[n];
  char *q = a + n;

  assert(q >= a);
  return 0;
}
