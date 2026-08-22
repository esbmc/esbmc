/* Rearranging the bounds check must not make it over-fire: every in-bounds
   byte offset still verifies, including the tight last one at
   data_sz - access_sz. Widths are fixed so the layout does not depend on the
   data model -- `long` is 32-bit on LLP64 targets, which moves every offset. */
#include <assert.h>
#include <stdint.h>

struct c
{
  char a;
  int32_t b;
  int64_t d;
};

int main(void)
{
  struct c s;
  s.a = 1;
  s.b = 2;
  s.d = 3;

  char *p = (char *)&s;

  assert(*(char *)(p + 0) == 1);
  assert(*(int32_t *)(p + 4) == 2);
  assert(*(int64_t *)(p + 8) == 3);
  assert(*(char *)(p + 15) == *((char *)&s.d + 7));

  return 0;
}
