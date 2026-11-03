#include <assert.h>
#include <stddef.h>
#include <string.h>
struct B { char c; _BitInt(65) v; };
struct S { char c; _BitInt(17) v; };
int main(void)
{
  struct B a, b;
  a.c = 1;
  a.v = 5;
  memcpy(&b, &a, sizeof a);
  assert(b.v == 5);
  assert(offsetof(struct B, v) == _Alignof(_BitInt(65)));
  assert(offsetof(struct S, v) == 4);
  struct B *p = &b;
  p->v = 7;
  assert(p->v == 7);
  return 0;
}
