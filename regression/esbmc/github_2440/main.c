#include <assert.h>

void foo(int m, int n, int x[m][n])
{
  x[1][0] = 1;
}

int main(void)
{
  int A[2][2] = {{0, 0}, {0, 0}};
  foo(2, 2, A);
  assert(A[1][0] == 1);
  return 0;
}
