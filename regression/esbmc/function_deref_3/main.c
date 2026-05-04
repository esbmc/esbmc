#include <assert.h>

void g(int x)
{
  assert(x == 5);
}

void trampo(void (*fp)())
{
  fp(5);
}

int main()
{
  trampo(g);
}
