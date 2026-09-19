// #7905: the cast is not dropped: all ones stays all ones.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned v4u __attribute__((__vector_size__(16)));

int main(void)
{
  v4i c = {-1, 0, 0, 0};
  v4u d = (v4u)c;
  assert(d[0] == 0);
  return 0;
}
