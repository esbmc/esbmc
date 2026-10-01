// #8008
#include <cassert>
bool nondet_bool();
int x = 1, y = 1;
int main()
{
  int *a = &y, *b = &y;
  bool c = nondet_bool();
  (c ? a : b) = &x;
  int t = x + 1;
  *a = 5;
  int u = x + 1;
  assert(u == 2);
  return 0;
}
