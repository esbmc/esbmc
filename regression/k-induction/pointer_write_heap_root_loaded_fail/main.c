// s is loaded from *pp, which the first loop writes, so symex cannot resolve
// it at the second loop's head: the step would drop the writes through s and
// prove a[0] < 3.
#include <stdlib.h>
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
int main()
{
  unsigned char *a = malloc(4);
  unsigned char *b = malloc(4);
  unsigned char **pp = malloc(sizeof *pp);
  a[0] = 0;
  *pp = a;
  unsigned char **cur = pp;
  while (__VERIFIER_nondet_int())
  {
    *cur = b;
    cur++;
    cur--;
  }
  unsigned char *s = *pp;
  for (;;)
  {
    *s = *s + 1;
    assert(a[0] < 3);
  }
}
