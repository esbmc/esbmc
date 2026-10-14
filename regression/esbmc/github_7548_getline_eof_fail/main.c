/* github #7548: end of file is reachable, so getline can return -1. The C
 * negative twin: nothing lets a caller assume a line was read. */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void)
{
  char *line = NULL;
  size_t cap = 0;
  ssize_t len = getline(&line, &cap, stdin);
  assert(len >= 0);
  free(line);
  return 0;
}
