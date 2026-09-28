// #7907: a lane of a vector read out of a packed struct straddles a member,
// which is reported as it is for a scalar read at that offset.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct __attribute__((packed)) rec
{
  char tag;
  int a, b, c, d;
};

int main(void)
{
  struct rec r = {1, 2, 3, 4, 5};
  v4i v = *(v4i *)&r;
  v4i w = v + v;
  assert(w[2] != 12345);
  return 0;
}
