/* #9383: an unnamed bit-field's type must not widen the layout past sizeof. */
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

struct A
{
  char c;
  int : 4;
};
struct S
{
  char c;
  int : 0;
  char d;
};
union U
{
  char c;
  int : 4;
};
struct L
{
  char c;
  long long : 0;
  char d;
};
struct W
{
  char c;
  struct A a;
  union U u;
  struct S s;
  struct L l;
};

#define STRIDE(T)                                                              \
  do                                                                           \
  {                                                                            \
    T x[2];                                                                    \
    assert((size_t)((char *)&x[1] - (char *)&x[0]) == sizeof(T));              \
  } while (0)

int main(void)
{
  STRIDE(struct A);
  STRIDE(struct S);
  STRIDE(union U);
  STRIDE(struct L);
  STRIDE(struct W);

  struct W w;
  assert((size_t)((char *)&w.a - (char *)&w) == offsetof(struct W, a));
  assert((size_t)((char *)&w.u - (char *)&w) == offsetof(struct W, u));
  assert((size_t)((char *)&w.s - (char *)&w) == offsetof(struct W, s));
  assert((size_t)((char *)&w.l - (char *)&w) == offsetof(struct W, l));

  struct A src[2] = {{1}, {2}}, dst[2];
  memcpy(dst, src, sizeof src);
  assert(dst[1].c == 2);

  struct A *p = malloc(3 * sizeof(struct A));
  if (p)
  {
    p[2].c = 1;
    free(p);
  }
  return 0;
}
