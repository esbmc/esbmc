#include <stdio.h>
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

/* Mirrors busybox bb_verror_msg: a runtime format string leaves the return
   value unbounded, so the asprintf failure path is reachable. POSIX leaves
   *strp undefined on failure, so returning without freeing is correct and
   must not be reported as a leak. */
static void log_msg(const char *fmt)
{
  char *msg;
  int used = asprintf(&msg, fmt, 1);
  if (used < 0)
    return;
  free(msg);
}

int main(void)
{
  const char *fmt = __VERIFIER_nondet_int() ? "%d" : "x%d";
  log_msg(fmt);
  return 0;
}
