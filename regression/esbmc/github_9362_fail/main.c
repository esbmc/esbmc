/* #9362: +0.0 == -0.0, so complex values differing only in the sign of a
   zero part are equal. */
#include <assert.h>

int main(void)
{
  double _Complex a = 0.0, b = -0.0;
  assert(a != b);
  return 0;
}
