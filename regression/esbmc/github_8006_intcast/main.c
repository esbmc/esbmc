// #8006
#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (!p) return 0;
  *p = 1;
  long addr = (long)p;
  int a = *p + 1;
  free((void *)addr);
  int b = *p + 1;
  return a + b;
}
