// k holds the result of id(a), whose goto_convert temporary is dead at the
// loop head. Havocing through the temporary would reach nothing, and the step
// would prove a[0] < 3.
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
unsigned char *id(unsigned char *x) { return x; }
int main()
{
  unsigned char *a = malloc(4);
  a[0] = 0;
  uintptr_t k;
  {
    k = (uintptr_t)id(a);
  }
  for (;;)
  {
    unsigned char *p = (unsigned char *)k;
    *p = *p + 1;
    assert(a[0] < 3);
  }
}
