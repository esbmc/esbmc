// #7907: a vector written through a pointer to an int array reaches every
// element of it.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  int buf[4] = {1, 2, 3, 4};
  *(v4i *)buf = (v4i){9, 8, 7, 6};
  assert(buf[3] == 4);
  return 0;
}
