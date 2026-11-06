#include <stdlib.h>

int main()
{
  char *p = malloc(10);
  if (!p)
    return 0;
  int s = 0;
  for (int i = 0; i <= 10; i++) s += p[i];
  return s;
}
