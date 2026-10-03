#include <cassert>
int main()
{
  int n = 3;
  using row = int[n];
  n = 5;
  row x;
  assert(sizeof(x) == 5 * sizeof(int)); /* row fixed at 3 */
  return 0;
}
