/* A constant vector combined with a vector that is not constant: each lane
 * takes the other vector's lane, not the whole vector. */
#include <assert.h>
typedef int v4i __attribute__((vector_size(16)));
int nondet_int();
int main()
{
  v4i a = {1, 2, 3, 4};
  v4i b;
  b[0] = nondet_int();
  b[1] = 0;
  b[2] = 5;
  b[3] = 0;
  v4i c = a + b;
  assert(c[2] == 3);
  return 0;
}
