// A zero innermost index with a non-zero enclosing row: the offset lives
// entirely in the linearised rows, so the fold has to emit it even though
// &a[..][0] alone would not be worth rewriting (#6778).
#include <assert.h>

int main(void)
{
  int a[2][3];
  int *p = &a[1][0];
  int *q = &a[0][0] + 3;

  assert(p == q);
  return 0;
}
