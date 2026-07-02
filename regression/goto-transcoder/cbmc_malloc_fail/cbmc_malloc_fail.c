#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (p)
  {
    p[1] = 5;
  }
  return 0;
}
