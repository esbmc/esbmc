#include <assert.h>
#include <stdio.h>
int main(void)
{
  int n = -1;
  printf("ab%n", &n);
  assert(n != 2);
  return 0;
}
