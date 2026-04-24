// Based on the follow-up minimal reproducer in issue #2180.
// Before the fix, SMT encoding crashed with a width-0 bit-vector error.
// After the fix, ESBMC reports the out-of-bounds read via the NULL source.
#include <string.h>

class a
{
};

struct b
{
  int c;
  a d;
};

int main()
{
  b e;
  e.c = 0;
  memmove(&e.d, 0, 1);
  return 0;
}
