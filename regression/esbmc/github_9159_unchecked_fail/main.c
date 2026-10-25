#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

/* asprintf may fail even for a constant format, leaving msg untouched, so
   writing through it unchecked is a NULL dereference (#9159). */
int main(void)
{
  char *msg = NULL;
  int used = asprintf(&msg, "hi");
  (void)used;
  msg[0] = 'x';
  free(msg);
  return 0;
}
