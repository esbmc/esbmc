#define _GNU_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* The busybox bb_verror_msg pattern from #9159: vasprintf allocates used + 1
   bytes, so writing the newline at msg[used] is in bounds. */
static void verror_msg(const char *fmt, ...)
{
  char *msg;
  va_list ap;
  va_start(ap, fmt);
  int used = vasprintf(&msg, fmt, ap);
  va_end(ap);
  if (used < 0)
    return;
  msg[used] = '\n';
  free(msg);
}

int main(void)
{
  verror_msg("%s: %d", "sync", 1);
  return 0;
}
