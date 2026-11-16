// s, which the loop moves only by offsets, holds what q held after a loop
// the step havocs q in, so symex cannot resolve s at the head: the step would
// drop the writes through s and prove a[0] < 3.
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
    s++;
    s--;
    assert(a[0] < 3);
  }
}
