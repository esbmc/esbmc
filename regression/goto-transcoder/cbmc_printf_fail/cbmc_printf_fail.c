#include <stdio.h>
int main()
{
  int x = 3;
  printf("x = %d\n", x);
  __CPROVER_assert(x == 4, "deliberately false");
  return 0;
}
