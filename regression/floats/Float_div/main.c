#include <stdio.h>

// Replace 1.6f to 2.5f to verification successful
#define X 2.5f

int main()
{
  float x = 1.0f;
  float x1 = x/X;

  while(x1 != x)
  {
    x = x1;
    x1 = x/X;
  }

  assert(x == 0);
  return 0;
}

