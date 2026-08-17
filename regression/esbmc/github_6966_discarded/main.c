#include <stdio.h>
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

/* Discarding the asprintf result leaves no return lvalue to guard the
   allocation with, so the buffer is allocated unconditionally and the caller
   owns it on every path (#6966). */
int main(void)
{
  const char *fmt = __VERIFIER_nondet_int() ? "%d" : "x%d";
  char *msg;
  asprintf(&msg, fmt, 1);
  free(msg);
  return 0;
}
