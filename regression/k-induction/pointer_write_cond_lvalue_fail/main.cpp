// The conditional l-value writes x or y through a pointer.
#include <cassert>
int nondet_int();
int x, y;
int main()
{
  int *p = &x;
  int *q = &y;
  x = 0;
  y = 0;
  while (1)
  {
    assert(x < 10);
    int c = nondet_int();
    (c ? *p : *q) = x + 1;
  }
}
