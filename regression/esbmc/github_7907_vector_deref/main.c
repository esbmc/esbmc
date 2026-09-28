// #7907: reading a whole vector through a pointer is an ordinary load.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i s = {1, 2, 3, 4};
  v4i *p = &s;

  v4i c = *p;
  assert(c[0] == 1 && c[1] == 2 && c[2] == 3 && c[3] == 4);

  // A lane read through the pointer, rather than the whole vector.
  assert((*p)[2] == 3);

  // Writing through the pointer is visible in the original.
  (*p)[1] = 9;
  assert(s[1] == 9);

  // A vector inside an aggregate, reached through a pointer to the aggregate.
  struct holder
  {
    int pad;
    v4i v;
  } h = {7, {5, 6, 7, 8}};
  struct holder *hp = &h;
  v4i g = hp->v;
  assert(g[3] == 8);
  assert(hp->pad == 7);
  return 0;
}
