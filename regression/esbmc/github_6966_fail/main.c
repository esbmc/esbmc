#include <stdio.h>
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

/* The success path still owns the buffer: dropping it is a genuine leak and
   must stay detected after the failure path stopped allocating (#6966). */
int main(void)
{
  const char *fmt = __VERIFIER_nondet_int() ? "%d" : "x%d";
  char *msg;
  int used = asprintf(&msg, fmt, 1);
  if (used < 0)
    return 0;
  return 0;
}
