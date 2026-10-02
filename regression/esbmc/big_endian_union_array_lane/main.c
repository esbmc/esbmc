#include <assert.h>

/* A union's arrays share their elements: a[1] is b[1] on either byte order. */
union U
{
  float f[4];
  int i[4];
  short a[4];
  short b[4];
};

int main()
{
  union U u = {0};
  u.f[1] = 1.0f;
  assert(u.i[1] == 0x3f800000);
  u.a[1] = 5;
  assert(u.b[1] == 5);
  return 0;
}
