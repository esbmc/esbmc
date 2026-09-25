/* The lowered offset is the real member displacement, not a nondet result: y
   sits one int past the start, so claiming zero must be caught. */
#include <assert.h>
#include <stddef.h>

struct s
{
  int x;
  int y;
};

int main(void)
{
  assert(offsetof(struct s, y) == 0);
  return 0;
}
