// Companion to github_723-empty: the zero-sized type-pun must not silently
// suppress genuine assertion failures reachable after it.
#include <assert.h>

typedef union
{
  struct
  {
  } x;
} t1;

typedef struct
{
  int y;
} t2;

t1 v1;
t2 v2;

int main()
{
  v2.y = 1;
  v1 = *(t1 *)&v2;
  assert(v2.y == 42);
  return 0;
}
