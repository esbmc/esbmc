#include <string.h>
#include <assert.h>

int main ()
{
  char str[] ="- This, a sample string.";
  char * pch;

  pch = strtok (str," ,.-");
  assert(!strcmp(pch, "This"));

  pch = strtok (NULL, " ,.-");
  assert(!strcmp(pch, "This"));

  pch = strtok (NULL, " ,.-");
  assert(!strcmp(pch, "sample"));

  pch = strtok (NULL, " ,.-");
  assert(!strcmp(pch, "string"));

  return 0;
}
