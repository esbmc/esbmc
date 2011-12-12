#include <assert.h>

int main()
{
  int x;
  int y;
  
  x= ({ y=1; 2; });

  assert(x==2);
  assert(y==1);

  return 0;
}
