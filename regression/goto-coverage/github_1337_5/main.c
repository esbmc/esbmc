#include <assert.h>

void loop1()
{
  for(int i = 0; i < 2; i++)
  {
    assert(0);
  }
}

void loop2()
{
  for(int i = 0; i < 6; i++)
  {
    assert(1);
  }
}

int main()

{
  switch(1)
  {
  case 1:
    loop1();
    break;
  case 2:
    loop2();
    break;
  default:;
  }
  return 0;
}