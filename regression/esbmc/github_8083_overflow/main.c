#include <assert.h>

/* The overflow builtins write their result through a pointer. */
int main()
{
  int r = 5;
  int a = nondet_int();
  __builtin_sadd_overflow(a, 1, &r);
  assert(r == 5);
  return 0;
}
