// #7897: a true lane is all ones (-1), not 1.
#include <assert.h>

typedef float v4f __attribute__((__vector_size__(16)));
typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4f a = {1, 2, 3, 4}, b = {1, 0, 3, 0};
  v4i m = a == b;
  assert(m[0] == 1);
  return 0;
}
