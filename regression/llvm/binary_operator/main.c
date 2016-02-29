#include <stdio.h>
#include <assert.h>

int main()
{
  int a = 2;

  int x = 1, y = 1, x1 = 1, x2 = 0;

  x2 = x2 + (+a);
  assert(x2 == 2);

  x2 = x2 + (-a);
  assert(x2 == 0);

  x2 = x2 + (!a);
  assert(x2 == 0);

  x2 = x2 + (~x2);
  assert(x2 == -1);

  x2 = x2 + (++a);
  assert(x2 == 2);

  x2 = x2 + (--a);
  assert(x2 == 4);

  x2 = x2 + (a++);
  assert(x2 == 6);

  x2 = x2 + (a--);
  assert(x2 == 9);

  float *x3 = (float*)&x2;
  x2 = *x3;
  assert(*x3 == 0.0f);

  int x4 = ((x2+x2)*(x2-x2))/x1;
  assert(x4 == 0);

  x = x + 1;
  assert(x == 2);

  x = x - 1;
  assert(x == 1);

  x = x / 1;
  assert(x == 1);

  x = x * x;
  assert(x == 1);

  x = x >> x;
  assert(x == 0);

  x = x << x;
  assert(x == 0);

  x = x % 1;
  assert(x == 0);

  x = x | x;
  assert(x == 0);

  x = x & x;
  assert(x == 0);

  x = x ^ x; 
  assert(x == 0);

  x = x > 1;
  assert(x == 0);

  x = x < y;
  assert(x == 1);

  x = x >= 1;
  assert(x == 1);

  x = x <= y;
  assert(x == 1);

  x = x == y;
  assert(x == 1);

  x = x != 1;
  assert(x == 0);

  x = x && 1;
  assert(x == 0);

  x = 6 && x;
  assert(x == 0);

  x = x || 1;
  assert(x == 1);

  return 0;
}
