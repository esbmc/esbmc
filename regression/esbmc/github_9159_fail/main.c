#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

/* The asprintf buffer holds used + 1 bytes: msg[used + 1] is past it (#9159). */
int main(void)
{
  char *msg;
  int used = asprintf(&msg, "%s", "hello");
  if (used < 0)
    return 0;
  msg[used + 1] = '\n';
  free(msg);
  return 0;
}
