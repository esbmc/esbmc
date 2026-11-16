// #9895: freeing a live heap object after a stack buffer it once pointed at
// went out of scope.
#include <stdlib.h>

int main()
{
  char *data;
  {
    char buf[4];
    data = buf;
  }
  data = malloc(4);
  free(data);
  return 0;
}
