#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

/* asprintf allocates used + 1 bytes, so msg[used] is in bounds (#9159). */
int main(void)
{
  char *msg;
  int used = asprintf(&msg, "%s", "hello");
  if (used < 0)
    return 0;
  msg[used] = '\n';
  free(msg);
  return 0;
}
