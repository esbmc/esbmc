/* goto-cc stores a union initialised through a member other than its first
 * as a byte_update of the zero union (#8238). */
#include <assert.h>

struct T
{
  int a;
  int b;
};

union U
{
  int i;
  float f;
};
union P
{
  long l;
  int *p;
};
union X
{
  struct T s;
  float f;
};
union Y
{
  struct T s;
  unsigned char c[3];
};

int x = 3;
union U gu = {.f = 1.5f};
union P gp = {.p = &x};
union X gx = {.f = 1.5f};
union Y gy = {.c = {1, 2, 3}};

int main(void)
{
  assert(gu.i == 0x3fc00001);
  assert(*gp.p == 4);
  assert(gx.s.a == 0);
  assert(gy.s.a == 0x010203);
  return 0;
}
