#include <stdlib.h>

int main()
{
  int *p = realloc(NULL, sizeof(int));
  *p = 42;
  return 0;
}
