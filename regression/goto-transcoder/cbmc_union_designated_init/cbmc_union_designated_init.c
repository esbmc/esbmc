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
  assert(gu.f == 1.5f && gu.i == 0x3fc00000);
  assert(*gp.p == 3);
  assert(gx.f == 1.5f && gx.s.b == 0);
  assert(gy.c[2] == 3 && gy.s.a == 0x030201);
  return 0;
}
