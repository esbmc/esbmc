// Exercises do_memcpy_expression's zero-byte early return:
// memcpy with size 0 must be a no-op and leave dst unchanged.
#include <assert.h>
#include <string.h>

int main()
{
  int dst = 42;
  int src = 100;

  memcpy(&dst, &src, 0);

  assert(dst == 42);
  return 0;
}
