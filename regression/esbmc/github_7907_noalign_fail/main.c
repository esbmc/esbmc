// #7907: under --no-align-check, a lane that cannot be built does not drop
// the other lanes of a vector written over a packed struct.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

struct __attribute__((packed)) rec
{
  char tag;
  int a, b, c, d;
};

int main(void)
{
  struct rec r = {5, 1, 2, 3, 4};
  *(v4i *)&r = (v4i){0, 0, 0, 0};
  assert(r.tag == 5);
  return 0;
}
