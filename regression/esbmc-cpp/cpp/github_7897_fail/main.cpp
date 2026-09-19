// #7897: a true lane is all ones (-1), not 1.
#include <cassert>

typedef float v4f __attribute__((__vector_size__(16)));
typedef int v4i __attribute__((__vector_size__(16)));

int main()
{
  v4f a = {1, 2, 3, 4};
  v4i lt = a < v4f{2, 2, 2, 2};
  assert(lt[0] == 1);
  return 0;
}
