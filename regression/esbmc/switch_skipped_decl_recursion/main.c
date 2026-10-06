#include <assert.h>

/* The switch jumps past y's declaration, but each call of f still has its
 * own y (C11 6.2.4p6), so the inner call does not overwrite the outer one. */
void f(int k)
{
  switch (k)
  {
    int y;
  case 0:
  case 1:
    y = k;
    if (k == 0)
      f(1);
    assert(y == k);
  }
}

int main()
{
  f(0);
}
