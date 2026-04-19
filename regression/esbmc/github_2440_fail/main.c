#include <assert.h>
#include <stdio.h>

void foo(int m, int n, int x[m][n])
{
  // Out-of-bounds access
  x[1][0] = 2;
}

int main(void)
{
  int C[1][1] = {{0}};
  foo(1, 1, C);
  assert(C[0][0] == 2);

  return 0;
}
