// The loop writes *s, where s holds what q held after a loop the step havocs
// q in. Havocing *s through an unresolved s reaches nothing, and the step
// would prove a[0] < 3.
#include <stdlib.h>
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
int main()
{
  unsigned char *a = malloc(4);
  unsigned char *b = malloc(4);
  a[0] = 0;
  unsigned char *q = a;
  while (__VERIFIER_nondet_int())
    q = b;
  unsigned char *s = q;
  for (;;)
  {
    *s = *s + 1;
    assert(a[0] < 3);
  }
}
