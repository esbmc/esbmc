// #7905: the bytes of a shuffled vector cast to bytes are the lanes' bytes.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned char v16c __attribute__((__vector_size__(16)));

int main(void)
{
  v4i a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
  v16c c = (v16c)__builtin_shufflevector(a, b, 4, 5, 6, 7);
  assert(c[1] == 6);
  return 0;
}
