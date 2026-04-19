#include <assert.h>

void foo(int m, int n, int x[m][n])
{
  x[1][0] = 1;
  x[0][0] = 2;
  x[m-1][n-1] = 3;
}

int main(void)
{
  int A[2][2] = {{0, 0}, {0, 0}};
  foo(2, 2, A);
  assert(A[1][0] == 1);
  assert(A[0][0] == 2);
  assert(A[1][1] == 3);

  int B[3][5] = {{0}};
  foo(3, 5, B);
  assert(B[1][0] == 1);
  assert(B[0][0] == 2);
  assert(B[2][4] == 3);

  return 0;
}
