// Negative counterpart: the condition selects `b`, so the copy holds 2 and the
// assertion must be violated.  Distributing the address over the arms must not
// make the wrong arm satisfiable.
#include <cassert>

struct P
{
  int v;
};

int main()
{
  P a{1}, b{2};
  int c = 5;

  P copy_init = (c < 1) ? a : b;
  assert(copy_init.v == 1);

  return 0;
}
