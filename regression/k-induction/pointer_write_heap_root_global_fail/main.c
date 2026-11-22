// G has one assignment, in setG, which runs twice before the loop. k holds
// G's first value (a); at the loop head G holds b.
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
unsigned char *G;
void setG(unsigned char *v) { G = v; }
int main()
{
  unsigned char *a = malloc(4);
  unsigned char *b = malloc(4);
  a[0] = 0;
  setG(a);
  uintptr_t k = (uintptr_t)G;
  setG(b);
  for (;;)
  {
    unsigned char *p = (unsigned char *)k;
    *p = *p + 1;
    assert(a[0] < 3);
  }
}
