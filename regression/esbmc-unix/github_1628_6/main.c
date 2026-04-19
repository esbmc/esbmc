#include <stdio.h>
struct Bar {};
int main(void)
{
  int x = 10;
  struct Bar b;
  printf("%d %s\n", x, b);
}
