#include <stdlib.h>
#include <assert.h>

// Negative companion to github_5337_sizeof_node: malloc(sizeof(struct T))
// allocates exactly one struct T, so indexing past it is out of bounds. This
// guards that the sizeof-node allocation is sized/typed correctly — if the
// size or type degraded, the out-of-bounds write below would be missed.

struct T
{
  int a;
  long b;
};

int main(void)
{
  struct T *p = malloc(sizeof(struct T));
  if (p)
  {
    p[1].a = 5; // out of bounds: only one element was allocated
    free(p);
  }
  return 0;
}
