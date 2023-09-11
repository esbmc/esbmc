#include <stdio.h>
#include <assert.h>
int main()
{
  long s = 100000;
  double ss= 10000.21;
  int x = printf("%d", s);
  int y = printf("%f\n", ss);
#ifndef _WIN32
  assert(x == 6);
  assert(y == 13);
  x+=1;
  y+=1;
#endif

}