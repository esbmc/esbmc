#include <assert.h>

int g(int v)
{
  return v;
}

// The call target `*(p + k)` makes `p + k` available again after `k = 1`
// without assigning its CSE symbol, so the symbol must not be reused (#7992).
int main()
{
  int arr[2] = {1, 1};
  int *p = arr;
  int k = 0;
  int x = *(p + k) + 1;
  k = 1;
  *(p + k) = g(5);
  int y = *(p + k) + 1;
  assert(y == 6);
}
