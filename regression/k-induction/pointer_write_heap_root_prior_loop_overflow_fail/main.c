// As pointer_write_heap_root_prior_loop_fail, for an overflow: dropping the
// writes through s would keep *a at INT_MAX - 3 and prove *a + 1 safe.
#include <stdlib.h>
#include <limits.h>
extern int __VERIFIER_nondet_int(void);
int main()
{
  int *a = malloc(sizeof(int));
  int *b = malloc(sizeof(int));
  *a = INT_MAX - 3;
  int *q = a;
  while (__VERIFIER_nondet_int())
    q = b;
  unsigned *s = (unsigned *)q;
  for (;;)
  {
    *s = *s + 1u;
    s++;
    s--;
    int y = *a + 1;
    (void)y;
  }
}
