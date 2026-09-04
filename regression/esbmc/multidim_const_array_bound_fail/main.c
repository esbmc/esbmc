// Anti-vacuity twin: the folded bound must still be the real one.
#include <assert.h>

int main(void)
{
  unsigned cnt = 0;
  int t[2][2] = {{4, 0}, {0, 0}};
  unsigned n = (unsigned)t[0][0];

  for (unsigned i = 0; i < n; i++)
    cnt++;

  assert(cnt == 5);
  return 0;
}
