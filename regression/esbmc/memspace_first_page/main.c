#include <assert.h>
#include <stdlib.h>

/* No object is placed on the first page, so an address made from a small
 * integer (e.g. an alignment used as a dangling pointer) never aliases one. */
char buf[16];

int main(void)
{
  char *heap = malloc(16);
  char *dangling = (char *)16;
  assert(buf != dangling);
  assert(heap != dangling);
  assert((unsigned long)buf != 4095);
  free(heap);
  return 0;
}
