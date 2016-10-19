#include <stdio.h>

// Replace 1.6 to 2.5 to verification successful
#define X 2.5

int main()
{
  double x = 1.0;
  double x1 = x/X;

  while(x1 != x)
  {
    x = x1;
    x1 = x/X;
  }

  assert(x == 0);
  return 0;
}

