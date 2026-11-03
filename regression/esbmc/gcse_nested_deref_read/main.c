#include <assert.h>

// The points-to analysis has no node for `**pp`, so it cannot name what it
// refers to, and a store to x must kill `**pp + 1` (#7992).
int main()
{
  int x = 1;
  int *p = &x;
  int **pp = &p;
  int a = **pp + 1;
  x = 5;
  int b = **pp + 1;
  assert(b == 6);
}
