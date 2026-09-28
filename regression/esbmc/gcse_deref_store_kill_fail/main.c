#include <assert.h>

int nondet_int();

// Reusing the stale `*p + 1` made b equal a.
int main()
{
  int x = nondet_int();
  int *p = &x;
  int a = *p + 1;
  *p = 3;
  int b = *p + 1;
  assert(b == a);
}
