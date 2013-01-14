#include <cassert>

int main()
{
  int a=0;

  while(int* x=new int)
  {
    ++a;

    if(a==5)
      break;
  }

  assert(a==5);

  return 0;
}
