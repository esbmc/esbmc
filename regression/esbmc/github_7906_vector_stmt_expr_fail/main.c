// #7906: the statement expression yields the vector itself, lanes intact.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i x = ({
    v4i t = {1, 2, 3, 4};
    t;
  });
  assert(x[3] == 1);
  return 0;
}
