#include <assert.h>
// A union defined inside a function body resolves its tag the same way.
int main()
{
  union U
  {
    unsigned i;
    unsigned char b[4];
  } u;
  u.i = 0x04030201;
  assert(u.b[0] == 0x01);
  return 0;
}