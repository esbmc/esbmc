// #7907: a vector read and written through a pointer of another type, and
// the other way round.
#include <assert.h>
#include <stdlib.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));

struct quad
{
  int a, b, c, d;
};

int main(void)
{
  int buf[4] = {1, 2, 3, 4};
  v4i *vp = (v4i *)buf;
  v4i c = *vp;
  assert(c[2] == 3);
  *vp = (v4i){9, 8, 7, 6};
  assert(buf[3] == 6);

  v4i s = {0x00020001, 2, 3, 4};
  int *ip = (int *)&s;
  assert(ip[1] == 2);
  ip[3] = 7;
  assert(s[3] == 7);

  v8s e = *(v8s *)&s;
  assert(e[0] == 1 && e[1] == 2);

  struct quad q = *(struct quad *)&s;
  assert(q.b == 2 && q.d == 7);

  v4i a[2] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  v4i *ap = a;
  v4i y = *(ap + 1);
  assert(ap[1][2] == 7 && y[3] == 8);

  unsigned char *m = malloc(16);
  if (m)
  {
    for (int k = 0; k < 16; k++)
      m[k] = k;
    assert((*(v4i *)m)[1] == 0x07060504);
    free(m);
  }
  return 0;
}
