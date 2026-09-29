/* github #7548: getline in C. The declaration was missing here too, but
 * clang's builtin table hid it; ssize_t and the model were absent in both. */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void)
{
  char *line = NULL;
  size_t cap = 0;

  ssize_t len = getline(&line, &cap, stdin);
  if (len >= 0)
  {
    assert(line != NULL);
    assert(cap > (size_t)len);
    assert(line[len] == '\0');
  }

  free(line);
  return 0;
}
