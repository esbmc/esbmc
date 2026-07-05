#include <stdlib.h>
int main()
{
  int *p = malloc(100);
  if (p)
  {
    *p = 5;
  }
  return 0;
}
