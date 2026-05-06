#include <assert.h>

// Regression for issue #4312-A nested case: zero-fill must descend
// into a nested struct, so the inner field is rendered for every
// witness even when the SMT model leaves it unconstrained.

struct inner
{
  int a;
  int b;
};

struct outer
{
  struct inner i;
  int c;
};

int main(void)
{
  struct outer o;
  if (o.i.a < -1 || o.i.a > 1)
    return 0;
  if (o.c < -1 || o.c > 1)
    return 0;
  // .i.b is unconstrained; the witness must still render it as 0.
  int absa = o.i.a < 0 ? -o.i.a : o.i.a;
  int absc = o.c < 0 ? -o.c : o.c;
  assert(absa + absc > 1);
  return 0;
}
