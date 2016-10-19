#include <math.h>
#include <stdio.h>
#include <assert.h>

int main()
{
  double x = 1.0/7.0;
  long long res = 0;

  for(int i = 1; x != 0; i++)
  {
    res += ((int)(x * 10) % 10) * (i * 10);
    x = (x * 10) - (int) x * 10;
  }

  assert(res == 56430);
  return 0;
}

