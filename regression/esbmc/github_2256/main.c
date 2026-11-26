/* #2256: strcmp over strings strcpy'd into malloc'd buffers unwound forever. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main()
{
  char *str1 = malloc(2);
  char *str2 = malloc(2);
  if (!str1 || !str2)
    return 0;
  strcpy(str1, "1");
  strcpy(str2, "1");
  assert(strcmp(str1, str2) == 0);
  return 0;
}
