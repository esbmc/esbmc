// k holds what t held before t = b, so the head's t is not the address the
// loop writes through. Havocing b's object instead, the step would prove
// a[0] < 3.
#include <stdlib.h>
#include <stdint.h>
extern void abort(void);
void reach_error() {}
void assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main()
{
  unsigned char *a = malloc(4);
  unsigned char *b = malloc(4);
  a[0] = 0;
  unsigned char *t = a;
  uintptr_t k = (uintptr_t)t;
  t = b;
  for (;;)
  {
    unsigned char *p = (unsigned char *)k;
    *p = *p + 1;
    assert(a[0] < 3);
  }
}
