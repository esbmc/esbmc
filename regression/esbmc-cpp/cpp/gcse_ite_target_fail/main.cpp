#include <cassert>

int nondet_int();

// Reusing the stale `x + 1` made b equal a.
int main()
{
  int x = 1, y = 1;
  int c = nondet_int();
  int a = x + 1;
  (c ? x : y) = 5;
  int b = x + 1;
  assert(b == a);
  return a;
}
