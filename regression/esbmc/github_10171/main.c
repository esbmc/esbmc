// #10171: an aligned array keeps its initializer.
#include <assert.h>
#include <string.h>
int a[8] __attribute__((aligned(16))) = {10, 20, 30, 30, 20, 10, 10, 20};
int main()
{
  int b[8];
  memcpy(b, a, sizeof(a));
  assert(b[0] == 10);
  _Alignas(16) int c[4] = {1, 2, 3, 4};
  assert(a[7] == 20 && c[3] == 4);
  return 0;
}
