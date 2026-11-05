// #9895: freeing a dead stack buffer is a bad free, not a dereference.
#include <stdlib.h>

int main()
{
  char *data;
  {
    char buf[4];
    data = buf;
  }
  free(data);
  return 0;
}
