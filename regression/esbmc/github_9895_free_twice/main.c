// #9895: a double free is a bad free, not a dereference.
#include <stdlib.h>

int main()
{
  char *p = malloc(4);
  free(p);
  free(p);
  return 0;
}
