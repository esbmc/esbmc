#include <assert.h>

struct S
{
  int a;
  int b;
};

struct S g;

int main(void)
{
  struct S l = {2, 3};
  g.a = l.a;
  assert(g.a == 2 && g.b == 0 && l.b == 3);
  return 0;
}
