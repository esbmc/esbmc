// #7905: reinterpret_cast to a reference to another vector type reinterprets
// the bits too.
#include <cassert>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned v4u __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));

int main()
{
  v4i c = {0x00020001, -1, 0, 0};
  v4u d = reinterpret_cast<v4u &>(c);
  assert(d[1] == 0xffffffffu);
  v8s h = reinterpret_cast<v8s &>(c);
  assert(h[0] == 1 && h[1] == 2);
  return 0;
}
