// #7907: under --no-align-check, the lanes of a vector written over a packed
// struct that can be built are written, as a scalar store there would be.
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
  assert(r.tag == 0);
  return 0;
}
