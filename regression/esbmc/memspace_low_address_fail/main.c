#include <assert.h>
#include <stdlib.h>

/* Without --reserve-first-page an object may start on the first page. */
int main(void)
{
  void *p = malloc(800);
  if (p != NULL && (unsigned long)p <= 2012)
    assert(0);
  return 0;
}
