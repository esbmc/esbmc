#include <stdlib.h>

int main()
{
  const char *s = "a"; // #10972
  char *p = malloc(2);
  if (p)
  {
    p[0] = s[0];
    free(p);
  }
  free(NULL);
  return 0;
}
