#include <assert.h>

/* Byte 16 of `struct S` is `s.v[1]`: a nonzero member offset composed with a
 * nonzero element offset. The value set got exactly this composition wrong --
 * each half worked alone and only the pair failed (R33,
 * docs/roadmap/goto-symex-verification-plan.md). dereferencet solves the same
 * descriptor-to-field-path problem with separate code, so pin that it agrees.
 * Reading an `int *` object through an `int **` is the object's own type, so
 * C11 6.5p7 is satisfied and this is not a strict-aliasing test. */
struct S
{
  long pad;
  int *v[2];
};

struct S s;
int g;

int main(void)
{
  int **pp = (int **)((char *)&s + 16);
  *pp = &g;
  assert(s.v[1] == &g);

  /* The read direction, and the two halves alone as controls. */
  s.v[0] = &g;
  int **p0 = (int **)((char *)&s + 8);
  assert(*p0 == &g);

  return 0;
}
