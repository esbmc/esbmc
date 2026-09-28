#include <stdio.h>
int main()
{
  int x = 3;
  printf("x = %d\n", x);
  __CPROVER_assert(x == 3, "printf does not disturb x");
  return 0;
}
