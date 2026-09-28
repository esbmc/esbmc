#include <assert.h>

union U
{
  int i;
  char c;
};

// Reusing the stale `u.c + 1` made y equal x.
int main()
{
  union U u;
  u.i = 1;
  int x = u.c + 1;
  u.i = 2;
  int y = u.c + 1;
  assert(y == x);
}
