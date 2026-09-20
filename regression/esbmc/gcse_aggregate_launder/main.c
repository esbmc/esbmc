// The store through p writes g, so the cached g + b cannot survive it. The
// address reaches p inside an aggregate initialiser of a pointer-free struct.
#include <assert.h>

int g = 1;

struct s
{
  unsigned long a;
};

int main()
{
  int b = 2;
  struct s s = {(unsigned long)&g};
  int *p = (int *)s.a;

  int x = g + b;
  *p = 100;
  int y = g + b;

  assert(y == 102);
  return 0;
}
