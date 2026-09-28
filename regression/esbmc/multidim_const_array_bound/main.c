// A loop bounded by a constant element of a multi-dimensional array. The
// value is statically determined, so the equation should reduce to nothing;
// without propagation the array and the index survive as constraints and the
// exit is never decided syntactically, so the same program with no --unwind
// runs forever.
#include <assert.h>

int main(void)
{
  unsigned cnt = 0;
  int t[2][2] = {{4, 0}, {0, 0}};
  unsigned n = (unsigned)t[0][0];

  for (unsigned i = 0; i < n; i++)
    cnt++;

  assert(cnt == 4);
  return 0;
}
