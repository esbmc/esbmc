#include <assert.h>

/* #10008: designated initialiser of a member of an anonymous struct. */
int main(void)
{
  union { struct { int s; int c; }; unsigned i; } v = { .s = 17 };
  assert(v.i == 17);
  assert(v.s == 17 && v.c == 0);
  return 0;
}
