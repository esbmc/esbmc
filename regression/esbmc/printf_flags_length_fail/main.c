#include <assert.h>
#include <stdio.h>

int main(void)
{
  int n = printf("%+d", 5);
  assert(n == 1);
  return 0;
}
