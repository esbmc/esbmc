// #10171: an alignas array keeps its initializer.
#include <cassert>
alignas(16) int a[4] = {1, 2, 3, 4};
int main()
{
  alignas(32) int b[2] = {5, 6};
  assert(a[0] == 1 && a[3] == 4 && b[1] == 6);
  return 0;
}
