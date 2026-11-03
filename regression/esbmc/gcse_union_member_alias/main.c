#include <assert.h>

union U
{
  int i;
  char c;
};

// `u.i = 2` also writes `u.c`, so `u.c + 1` must be recomputed.
int main()
{
  union U u;
  u.i = 1;
  int x = u.c + 1;
  u.i = 2;
  int y = u.c + 1;
  assert(y == 3);
}
