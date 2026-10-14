#include <assert.h>

// q points to x, so `*q + 1` must be recomputed after x is written, both
// through the alias p and directly.
int main()
{
  int x = 1;
  int *p = &x, *q = &x;
  int a = *q + 1;
  *p = 5;
  int b = *q + 1;
  int c = *q + 1;
  x = 7;
  int d = *q + 1;
  assert(b == a || d == c);
}
