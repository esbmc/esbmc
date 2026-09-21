// #7905: a shuffle or conversion used inside an expression, such as a cast, has
// the vector type it returns.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned char v16c __attribute__((__vector_size__(16)));
typedef float v4f __attribute__((__vector_size__(16)));

int main(void)
{
  v4i a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
  v4i x = __builtin_shufflevector(a, b, 0, 4, 1, 5) + a;
  assert(x[1] == 7 && x[3] == 10);
  v16c c = (v16c)__builtin_shufflevector(a, b, 4, 5, 6, 7);
  assert(c[0] == 5 && c[1] == 0 && c[4] == 6);
  v4f f = __builtin_convertvector(a, v4f) * 2.0f;
  assert(f[3] == 8.0f);
  return 0;
}
