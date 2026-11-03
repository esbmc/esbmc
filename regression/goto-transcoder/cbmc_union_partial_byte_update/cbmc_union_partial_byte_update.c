/* A bitfield member of a union whose first member is a pointer is stored as a
 * byte_update of a 3-bit value over a null pointer, which ESBMC's byte-wise
 * byte_update cannot express; it is refused rather than misencoded (#8238). */
#include <assert.h>

union B
{
  int *p;
  unsigned b : 3;
};

union B gb = {.b = 5};

int main(void)
{
  assert(gb.b == 5);
  return 0;
}
