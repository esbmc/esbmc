// #7907: a vector member of a struct, read and written through a pointer to
// it, sits at its own alignment.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct S
{
  int x;
  v4i v;
};

int main(void)
{
  struct S s = {7, {1, 2, 3, 4}};
  v4i *p = &s.v;
  v4i c = *p;
  assert(c[1] == 2);
  *p = (v4i){0, 0, 0, 0};
  assert(s.v[3] == 0 && s.x == 7);
  return 0;
}
