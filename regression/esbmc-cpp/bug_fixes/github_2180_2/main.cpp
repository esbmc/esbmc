#include <string.h>
#include <cassert>

class empty
{
};

struct b
{
  int c;
  empty d;
};

int main()
{
  b e;
  e.c = 0;

  int src = 42;
  memmove(&e.c, &src, sizeof(int));

  assert(e.c == 42);
  return 0;
}
