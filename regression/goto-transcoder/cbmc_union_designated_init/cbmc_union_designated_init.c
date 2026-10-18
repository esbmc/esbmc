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
union Q
{
  struct T s;
  int *p;
};
struct C
{
  char a;
};
union R
{
  struct T s;
  struct C q[2];
};
union B
{
  struct T s;
  unsigned b : 3;
};
/* A null pointer first member keeps these off the union-constant path. */
union N
{
  int *p;
  long l;
};
union M
{
  int *p;
  long l;
  unsigned char c[3];
};
union F
{
  float f;
  int i;
};
union K
{
  int *p;
  struct C q[2];
};

int x = 3;
union U gu = {.f = 1.5f};
union P gp = {.p = &x};
union X gx = {.f = 1.5f};
union Y gy = {.c = {1, 2, 3}};
union Q gq = {.p = &x};
union R gr = {.q = {{1}, {2}}};
union B gb = {.b = 5};
union N gn = {.l = 0x0102030405060708};
union M gm = {.c = {1, 2, 3}};
union F gf = {.i = 1};
union K gk = {.q = {{1}, {2}}};

int main(void)
{
  assert(gu.f == 1.5f && gu.i == 0x3fc00000);
  assert(*gp.p == 3);
  assert(gx.f == 1.5f && gx.s.b == 0);
  assert(gy.c[2] == 3 && gy.s.a == 0x030201);
  assert(*gq.p == 3);
  assert(gr.q[1].a == 2 && gr.s.a == 0x0201);
  assert(gb.b == 5 && gb.s.a == 5);
  assert(gn.l == 0x0102030405060708);
  assert(gm.c[2] == 3 && (gm.l & 0xffffff) == 0x030201);
  assert(gk.q[1].a == 2);
  assert(gf.i == 1 && gf.f != 0);
  return 0;
}
