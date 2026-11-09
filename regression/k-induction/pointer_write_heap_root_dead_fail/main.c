// k holds the address in t, which is dead at the loop head. Havocing through
// t would reach nothing, and the step would prove a[0] < 3.
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
int main()
{
  unsigned char *a = malloc(4);
  a[0] = 0;
  uintptr_t k;
  {
    unsigned char *t = a;
    k = (uintptr_t)t;
  }
  for (;;)
  {
    unsigned char *p = (unsigned char *)k;
    *p = *p + 1;
    assert(a[0] < 3);
  }
}
