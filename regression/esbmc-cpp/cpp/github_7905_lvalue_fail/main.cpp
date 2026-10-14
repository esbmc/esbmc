// #7905: the reference cast is not dropped: the lanes read back are the bits.
#include <cassert>

typedef int v4i __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));

int main()
{
  v4i c = {0x00020001, -1, 0, 0};
  v8s h = reinterpret_cast<v8s &>(c);
  assert(h[1] != 2);
  return 0;
}
