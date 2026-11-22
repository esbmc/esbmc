#include <stdio.h>
int f(void) { return 0; }
int main()
{
  sscanf("1", "%d", (int *)(void *)f);
  return 0;
}
