#include <cassert>

int main()
{
  int a=0;

  if(int x=5)
    a=5;

  assert(a==5);

  return 0;
}
