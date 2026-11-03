// memchr returns a pointer into buf through an intrinsic symex runs, not a
// nondet value: taking it for one would leave what d points to unhavoced and
// prove d[4] == 0.
#include <stdlib.h>
#include <string.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

void mark(char *buf)
{
  char *q = memchr(buf, 0, 8);
  if (q)
    *q = 1;
}

int main()
{
  char *d = malloc(8);
  memset(d, 0, 8);
  for (;;)
  {
    mark(d);
    __VERIFIER_assert(d[4] == 0);
  }
}
