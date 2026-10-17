#include <assert.h>

/* The write reached b[2] on a big-endian target, so this assertion held. */
union U
{
  short a[4];
  short b[4];
};

int main()
{
  union U u = {0};
  u.a[1] = 5;
  assert(u.b[1] != 5);
  return 0;
}
