/* github #7548: the bytes getline returns are the stream's, so none of the
 * caller's previous contents may survive into them. Modelling the buffer as
 * reused proved this assertion, and with it that two different input lines
 * start with the same byte. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void)
{
  char *line = malloc(100);
  size_t cap = 100;
  if (line == NULL)
    return 0;
  memset(line, 'a', cap);

  ssize_t len = getline(&line, &cap, stdin);
  if (len > 0 && (size_t)len < 100)
    assert(line[0] == 'a');

  free(line);
  return 0;
}
