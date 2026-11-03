#include <cassert>
int main()
{
  int n = 3;
  using row = int[n];
  n = 5;
  row x;
  assert(sizeof(x) == 3 * sizeof(int)); /* row fixed at 3 (C11 6.7.8p3) */
  return 0;
}
