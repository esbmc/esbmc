#include <stdio.h>
const int c = 3;
int main()
{
  sscanf("5", "%d", (int *)&c);
  return 0;
}
