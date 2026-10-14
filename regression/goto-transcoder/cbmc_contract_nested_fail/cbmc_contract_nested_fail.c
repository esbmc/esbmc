#include <stdlib.h>

int leaf(int x)
  __CPROVER_requires(x >= 0 && x < 100)
  __CPROVER_ensures(__CPROVER_return_value == x + 1)
{
  return x + 1;
}

int caller(int y)
  __CPROVER_requires(y >= 0 && y < 50)
  __CPROVER_ensures(__CPROVER_return_value == y + 2)
{
  return leaf(y);
}

int main() { return 0; }
