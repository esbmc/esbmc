/* The realloc half of R38: the cap joins realloc's failure condition, so an
   over-cap request returns NULL and never lays out an object whose upper
   offsets read negative. See ptr_rel_huge_object for the malloc spelling. The
   width is fixed: `unsigned long` is 32-bit on LLP64 and never reaches 2^63. */
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();

  char *p = malloc(1);
  if (!p)
    return 0;

  p = realloc(p, n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
