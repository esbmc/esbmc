#include <assert.h>
#include <stdio.h>

struct s
{
  int a;
  int b;
} g = {1, 2};

int main()
{
  sscanf("5", "%d", &g.b);
  assert(g.a == 1);
  return 0;
}
