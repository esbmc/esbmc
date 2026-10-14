// #7907: a vector read out of a packed struct whose members line up with its
// lanes.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct __attribute__((packed)) rec
{
  int a, b, c, d;
  char tag;
};

int main(void)
{
  struct rec r = {1, 2, 3, 4, 5};
  v4i v = *(v4i *)&r;
  v4i w = v + v;
  assert(w[2] == 6);
  return 0;
}
