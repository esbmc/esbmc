#include <assert.h>

int funcA()
{
  return 2;
}

int main()
{
  int a,b,c;

  a=funcA();
  b=funcA;
  c=a*b;

  assert(c!=a*b);
}
