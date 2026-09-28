#include <assert.h>

struct S
{
  int v;
};

// The points-to analysis names the target of `&s->v` by the expression `*s`,
// so the store through r must be treated as writing an unknown object and
// kill `n.v + 1` (#7992).
int main()
{
  struct S n;
  n.v = 1;
  struct S *s = &n;
  int *r = &s->v;
  int a = n.v + 1;
  *r = 5;
  int b = n.v + 1;
  assert(b == a);
}
