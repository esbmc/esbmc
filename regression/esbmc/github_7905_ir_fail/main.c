// #7905: under --ir a vector bitcast between lanes of one width converts each
// lane's value: the lanes keep their order.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned v4u __attribute__((__vector_size__(16)));

int nondet_int(void);

int main(void)
{
  v4i c = {nondet_int(), 1, 2, 3};
  v4u d = (v4u)c;
  assert(d[3] == 2u);
  return 0;
}
