#include <assert.h>
void g(int x, int y)
{
  assert(0);
}
void trampo(void (*g)())
{
  g(5);
}
int main()
{
  trampo(g);
}
